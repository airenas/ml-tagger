use std::{fs::File, io::BufReader, sync::Arc};

use anyhow::Ok;
use async_trait::async_trait;
use finalfusion::io::ReadEmbeddings;
use finalfusion::prelude::MmapEmbeddings;
use finalfusion::{prelude::Embeddings, storage::StorageWrap, vocab::VocabWrap};

use moka::future::Cache;

use crate::{
    handlers::data::{Processor, WorkContext},
    utils::perf::PerfLogger,
};

const EMAIL_WORD: &str = "<email>";
const URL_WORD: &str = "<url>";

pub struct FinalFusionWrapper {
    embeds: Embeddings<VocabWrap, StorageWrap>,
    cache: Cache<String, Arc<Vec<f32>>>,
}

impl FinalFusionWrapper {
    pub fn new(
        file: &str,
        cache: Cache<String, Arc<Vec<f32>>>,
    ) -> anyhow::Result<FinalFusionWrapper> {
        let _perf_log = PerfLogger::new("finalfusion loader");
        let embeds = if let Some(actual_path) = file.strip_prefix("mmap:") {
            let mut reader = BufReader::new(File::open(actual_path)?);
            tracing::info!(file = actual_path, "loading FinalFusion as MemMap file");
            Embeddings::mmap_embeddings(&mut reader)?
        } else {
            let mut reader = BufReader::new(File::open(file)?);
            tracing::info!(file, "loading FinalFusion into memory");
            Embeddings::<VocabWrap, StorageWrap>::read_embeddings(&mut reader)?
        };

        tracing::info!(dims = embeds.dims(), "loaded");
        let res = FinalFusionWrapper { embeds, cache };
        Ok(res)
    }
    pub fn dims(&self) -> usize {
        self.embeds.dims()
    }
}

#[async_trait]
impl Processor for FinalFusionWrapper {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("finalfusion embeddings");
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                if word_info.is_word {
                    let w =
                        match word_info.kind {
                            crate::handlers::data::WordKind::Email => EMAIL_WORD,
                            crate::handlers::data::WordKind::Url => URL_WORD,
                            _ => &word_info.w,
                        };
                    let emb = self.cache.get(w).await;
                    if let Some(val) = emb {
                        tracing::trace!(word = w, "in emb cache");
                        word_info.embeddings = Some(val);
                    } else {
                        tracing::trace!(word = w, "calc embeddings");
                        let embedding = self
                            .embeds
                            .embedding_with_norm(w)
                            .ok_or_else(|| anyhow::anyhow!("missing word: {}", w))?
                            .embedding
                            .to_owned()
                            .to_vec();
                        let val = Arc::new(embedding);
                        word_info.embeddings = Some(val.clone());
                        self.cache.insert(w.to_string(), val).await;
                    }
                }
            }
        }
        Ok(())
    }
}
