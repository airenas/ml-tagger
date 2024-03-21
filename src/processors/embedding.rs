use anyhow::Ok;
use fasttext::FastText;

use crate::{
    handlers::data::{Processor, WorkContext},
    utils::PerfLogger,
};

pub struct FastTextWrapper {
    model: FastText,
}

impl FastTextWrapper {
    pub fn new(file: &str) -> anyhow::Result<FastTextWrapper> {
        let _perf_log = PerfLogger::new("fast text loader");
        let mut model = FastText::new();
        log::info!("Loading FastText from {}", file);
        model.load_model(file).map_err(anyhow::Error::msg)?;
        log::info!("Loaded FastText dim {}", model.get_dimension(),);
        let res = FastTextWrapper { model };
        Ok(res)
    }
}

impl Processor for FastTextWrapper {
    fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("fast text embeddigs");
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                let embedding = self
                    .model
                    .get_word_vector(&word_info.w)
                    .map_err(anyhow::Error::msg)?;
                word_info.embeddings = Some(embedding);
            }
        }
        Ok(())
    }
}
