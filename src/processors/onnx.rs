use std::time::Instant;

use anyhow::Ok;
use onnxruntime_ng::{
    environment::Environment, ndarray, tensor::OrtOwnedTensor, GraphOptimizationLevel, LoggingLevel,
};

use crate::handlers::data::{Processor, WorkContext};

pub struct OnnxWrapper {
    environment: Environment,
    file: String,
    // session: Option<Arc<Session<'a>>>,
}

impl OnnxWrapper {
    pub fn new(file: &str) -> anyhow::Result<OnnxWrapper> {
        // let before = Instant::now();
        let environment = Environment::builder()
            .with_name("onnx")
            .with_log_level(LoggingLevel::Verbose)
            .build()?;
        let res = OnnxWrapper {
            environment,
            file: file.to_string(),
        };
        Ok(res)
    }
}

impl Processor for OnnxWrapper {
    fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let before = Instant::now();
        log::debug!("Loading ONNX model from {}", self.file);
        let mut session = self
            .environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::Basic)?
            .with_number_threads(4)?
            .with_model_from_file(self.file.as_str())?;
        log::debug!("onnx loaded in {:.2?}", before.elapsed());

        let before = Instant::now();
        for sent in ctx.sentences.iter_mut() {
            let mut combined_data: Vec<f32> = Vec::new();
            let mut cw = 0;
            for word_info in sent.iter_mut() {
                match &word_info.embeddings {
                    Some(emb) => combined_data.extend(emb),
                    None => {}
                }
                cw += 1;
            }
            let array = ndarray::Array::from_vec(combined_data)
                .into_shape((1, cw, 150))
                .unwrap();
            let input_tensor = vec![array];
            let outputs: Vec<OrtOwnedTensor<i32, _>> = session.run(input_tensor)?;
            let mut output_values: Vec<i32> = Vec::new();
            for tensor in outputs {
                let tensor_values: Vec<i32> = tensor.iter().copied().collect();
                // Add values to the output vector
                output_values.extend(tensor_values);
            }

            for (i, word_info) in sent.iter_mut().enumerate() {
                word_info.predicted = Some(output_values[i])
            }
        }

        log::debug!("Done onnx in {:.2?}", before.elapsed());
        Ok(())
    }
}
