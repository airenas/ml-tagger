use std::sync::Arc;

use anyhow::Ok;
use async_trait::async_trait;
use fasttext::FastText;
use moka::future::Cache;

use crate::{
    handlers::data::{Processor, WorkContext},
    utils::perf::PerfLogger,
};

pub struct FastTextWrapper {
    model: FastText,
    cache: Cache<String, Arc<Vec<f32>>>,
}

impl FastTextWrapper {
    pub fn new(file: &str, cache: Cache<String, Arc<Vec<f32>>>) -> anyhow::Result<FastTextWrapper> {
        let _perf_log = PerfLogger::new("fast text loader");
        let mut model = FastText::new();
        log::info!("Loading FastText from {}", file);
        model.load_model(file).map_err(anyhow::Error::msg)?;
        log::info!("Loaded FastText dim {}", model.get_dimension(),);
        let res = FastTextWrapper { model, cache };
        Ok(res)
    }
}

#[async_trait]
impl Processor for FastTextWrapper {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("fast text embeddings");
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                if word_info.is_word {
                    let w = &word_info.w;
                    let emb = self.cache.get(w).await;
                    if let Some(val) = emb {
                        log::debug!("in emb cache: {w}");
                        let new_value = Arc::clone(&val);
                        let extracted = (*new_value).clone();
                        word_info.embeddings = Some(extracted);
                    } else {
                        let embedding = self
                            .model
                            .get_word_vector(&word_info.w)
                            .map_err(anyhow::Error::msg)?;
                        word_info.embeddings = Some(embedding.clone());
                        self.cache.insert(w.clone(), Arc::new(embedding)).await;
                    }
                }
            }
        }
        Ok(())
    }
}
