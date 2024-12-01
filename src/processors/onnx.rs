use ndarray::Array;
use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreMLExecutionProvider,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::env;

use async_trait::async_trait;

use crate::handlers::data::{Processor, WorkContext};
use crate::utils::perf::PerfLogger;

pub struct OnnxWrapper {
    // environment: Environment,
    // model_bytes: Vec<u8>,
    threads: u16,
    model: Session,
}

impl OnnxWrapper {
    pub fn new(file_str: &str, threads: u16) -> anyhow::Result<OnnxWrapper> {
        match env::var("LD_LIBRARY_PATH") {
            Ok(value) => log::info!("LD_LIBRARY_PATH: {}", value),
            Err(_) => log::warn!("LD_LIBRARY_PATH env var not found"),
        };
        let _perf_log = PerfLogger::new("onnx loader");

        // ort::init()
        //     .with_execution_providers([CPUExecutionProvider::default().build().error_on_failure()])
        //     .commit()?;
        // ort::init()
        //     .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
        //     .commit()?;
        log::info!("Loading ONNX model from {}", file_str);
        // let model = Session::builder()?.commit_from_file(file_str)?;
        let model = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default()
                // .with_cuda_graph()
                .build()
                .error_on_failure()])?
            // .with_execution_providers([CoreMLExecutionProvider::default()
            // .with_cpu_only()
            // this model uses control flow operators, so enable CoreML on subgraphs too
            // .with_subgraphs()
            // only use the ANE as the CoreML CPU implementation is super slow for this model
            // .with_ane_only()
            // .build()])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            // .with_intra_threads(12)?
            .commit_from_file(file_str)?;

        // let environment = Environment::builder()
        //     .with_name("onnx")
        //     .with_log_level(LoggingLevel::Verbose)
        //     .build()?;
        // let mut file = File::open(file_str)?;
        // let mut buffer = Vec::new();
        // file.read_to_end(&mut buffer)?;

        show_info(&model)?;

        // log::info!("ONNX threads {threads}");
        let res = OnnxWrapper {
            //environment,
            // model_bytes: buffer,
            threads,
            model,
        };
        Ok(res)
    }
}

fn show_info(model: &Session) -> anyhow::Result<()> {
    let meta = model.metadata()?;
    if let Ok(x) = meta.name() {
        log::info!("Name: {x}");
    }
    if let Ok(x) = meta.description() {
        log::info!("Description: {x}");
    }
    if let Ok(x) = meta.producer() {
        log::info!("Produced by {x}");
    }

    log::info!("Inputs:");
    for (i, input) in model.inputs.iter().enumerate() {
        log::info!("    {i} {}: {}", input.name, input.input_type);
    }
    log::info!("Outputs:");
    for (i, output) in model.outputs.iter().enumerate() {
        log::info!("    {i} {}: {}", output.name, output.output_type);
    }
    Ok(())
}

#[async_trait]
impl Processor for OnnxWrapper {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("onnx");
        // let _inner_perf_log = PerfLogger::new("onnx load from mem");
        // let mut session = self
        //     .environment
        //     .new_session_builder()?
        //     .use_cuda(0)?
        //     .with_optimization_level(GraphOptimizationLevel::DisableAll)?
        //     .with_number_threads(self.threads)?
        //     .with_model_from_memory(self.model_bytes.clone())?;

        // std::mem::drop(_inner_perf_log);
        let _inner_perf_log = PerfLogger::new("onnx run");
        for sent in ctx.sentences.iter_mut() {
            let mut combined_data: Vec<f32> = Vec::new();
            let mut cw = 0;
            for word_info in sent.iter_mut() {
                if word_info.is_word {
                    match &word_info.embeddings {
                        Some(emb) => combined_data.extend(emb.iter()),
                        None => {}
                    }
                    cw += 1;
                }
            }
            let input_tensor = Array::from_shape_vec((1, cw, 150), combined_data)?;
            let outputs = self.model.run(ort::inputs![input_tensor]?)?;
            let shape = outputs[0].shape();
            log::trace!("Output shape: {:?}", shape);
            let output_tensors = outputs[0].try_extract_tensor::<i32>()?;
            let output_values: Vec<i32> = output_tensors.iter().map(|i| *i).collect();
            //    log::info!("output_values: {:?}", output_values);
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
