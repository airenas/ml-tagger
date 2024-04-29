use std::env;
use std::fs::File;
use std::io::Read;

use async_trait::async_trait;
use onnxruntime_ng::{
    environment::Environment, ndarray, tensor::OrtOwnedTensor, GraphOptimizationLevel, LoggingLevel,
};

use crate::handlers::data::{Processor, WorkContext};
use crate::utils::perf::PerfLogger;

pub struct OnnxWrapper {
    environment: Environment,
    model_bytes: Vec<u8>,
    threads: i16,
}

impl OnnxWrapper {
    pub fn new(file_str: &str, threads: i16) -> anyhow::Result<OnnxWrapper> {
        match env::var("LD_LIBRARY_PATH") {
            Ok(value) => log::info!("LD_LIBRARY_PATH: {}", value),
            Err(_) => log::warn!("LD_LIBRARY_PATH env var not found"),
        };
        let _perf_log = PerfLogger::new("onnx loader");
        let environment = Environment::builder()
            .with_name("onnx")
            .with_log_level(LoggingLevel::Verbose)
            .build()?;
        log::info!("Loading ONNX model from {}", file_str);
        let mut file = File::open(file_str)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        log::info!("ONNX threads {threads}");
        let res = OnnxWrapper {
            environment,
            model_bytes: buffer,
            threads,
        };
        Ok(res)
    }
}

#[async_trait]
impl Processor for OnnxWrapper {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("onnx");
        let _inner_perf_log = PerfLogger::new("onnx load from mem");
        let mut session = self
            .environment
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::DisableAll)?
            .with_number_threads(self.threads)?
            .with_model_from_memory(self.model_bytes.clone())?;

        std::mem::drop(_inner_perf_log);
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
