use std::time::Instant;

use anyhow::Ok;
use fasttext::FastText;

use crate::handlers::data::{Processor, WorkContext};

pub struct FastTextWrapper {
    model: FastText,
}

impl FastTextWrapper {
    pub fn new(file: &str) -> anyhow::Result<FastTextWrapper> {
        let before = Instant::now();
        let mut model = FastText::new();
        log::debug!("Loading FastText from {}", file);
        model.load_model(file).map_err(anyhow::Error::msg)?;
        log::debug!(
            "Loaded FastText dim {} in {:.2?}",
            model.get_dimension(),
            before.elapsed()
        );
        let res = FastTextWrapper { model };
        Ok(res)
    }
}

impl Processor for FastTextWrapper {
    fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let before = Instant::now();
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                let embedding = self
                    .model
                    .get_word_vector(&word_info.w)
                    .map_err(anyhow::Error::msg)?;
                word_info.embeddings = Some(embedding);
            }
        }
        log::debug!("Done embedding in {:.2?}", before.elapsed());
        Ok(())
    }
}
