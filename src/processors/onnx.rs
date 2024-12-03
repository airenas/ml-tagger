use ndarray::Array;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::env;

use async_trait::async_trait;

use crate::handlers::data::{Processor, WorkContext};
use crate::utils::perf::PerfLogger;

pub struct OnnxWrapper {
    emb_dim: usize,
    model: Session,
}

impl OnnxWrapper {
    pub fn new(file_str: &str, threads: u16, emb_dim: usize) -> anyhow::Result<OnnxWrapper> {
        match env::var("LD_LIBRARY_PATH") {
            Ok(value) => log::info!("LD_LIBRARY_PATH: {}", value),
            Err(_) => log::warn!("LD_LIBRARY_PATH env var not found"),
        };
        let _perf_log = PerfLogger::new("onnx loader");
        // ort::init()
        //     .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
        //     .commit()?;
        tracing::info!(
            onnx_tgreads = threads,
            file = file_str,
            "loading ONNX model"
        );
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_inter_threads(threads as usize)?
            .commit_from_file(file_str)?;

        show_info(&model)?;
        let model_input_dim = get_dim(&model)?;
        if emb_dim != model_input_dim {
            return Err(anyhow::anyhow!(
                "Model input dimension ({}) does not match embedding dim ({})",
                model_input_dim,
                emb_dim
            ));
        }

        let res = OnnxWrapper {
            model,
            emb_dim,
        };
        Ok(res)
    }
}

fn show_info(model: &Session) -> anyhow::Result<()> {
    let meta = model.metadata()?;
    if let Ok(x) = meta.name() {
        tracing::info!(name = x, "model");
    }
    if let Ok(x) = meta.description() {
        tracing::info!(description = x, "model");
    }
    if let Ok(x) = meta.producer() {
        tracing::info!(by = x, "model");
    }

    for (i, input) in model.inputs.iter().enumerate() {
        tracing::info!(
            i,
            name = input.name,
            "type" = format!("{}", input.input_type),
            "input"
        );
    }
    for (i, output) in model.outputs.iter().enumerate() {
        tracing::info!(
            i,
            name = output.name,
            "type" = format!("{}", output.output_type),
            "output"
        );
    }
    Ok(())
}

fn get_dim(model: &Session) -> anyhow::Result<usize> {
    if model.inputs.len() != 1 {
        return Err(anyhow::anyhow!("Expected 1 input tensor"));
    }
    let input = &model.inputs[0];
    if input.input_type.is_tensor() {
        let shape = input
            .input_type
            .tensor_dimensions()
            .ok_or_else(|| anyhow::anyhow!("Input not a tensor"))?;
        if shape.len() != 3 {
            return Err(anyhow::anyhow!("Expected 3D tensor"));
        }
        Ok(shape[2] as usize)
    } else {
        Err(anyhow::anyhow!("Expected tensor input"))
    }
}

#[async_trait]
impl Processor for OnnxWrapper {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("onnx");
        for sent in ctx.sentences.iter_mut() {
            let count = sent.iter().filter(|x| x.is_word).count();
            let mut combined_data: Vec<f32> = Vec::with_capacity(count * self.emb_dim);
            for word_info in sent.iter() {
                if word_info.is_word {
                    if let Some(emb) = &word_info.embeddings { 
                        combined_data.extend(emb.iter()) 
                    } else {
                        return Err(anyhow::anyhow!("Missing embeddings"));
                    }
                }
            }
            let input_tensor = Array::from_shape_vec((1, count, self.emb_dim), combined_data)?;
            let outputs = self.model.run(ort::inputs![input_tensor]?)?;
            let shape = outputs[0].shape();
            log::trace!("Output shape: {:?}", shape);
            let output_tensors = outputs[0].try_extract_tensor::<i32>()?;
            let output_values: Vec<i32> = output_tensors.iter().copied().collect();
            let mut i = 0;
            for word_info in sent.iter_mut() {
                if word_info.is_word {
                    word_info.predicted = Some(output_values[i]);
                    i += 1;
                }
            }
        }
        Ok(())
    }
}
