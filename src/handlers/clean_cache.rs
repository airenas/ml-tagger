use std::{borrow::Cow, sync::Arc};

use axum::extract::State;
use moka::future::Cache;

use super::{data::WorkMI, error::ApiError};

pub struct CacheData {
    pub lemma_cache: Cache<String, Arc<Vec<WorkMI>>>,
    pub embeddings_cache: Cache<String, Arc<Vec<f32>>>,
}

pub async fn handler(
    State(caches): axum::extract::State<Arc<CacheData>>,
) -> Result<Cow<'static, str>, ApiError> {
    log::debug!("clean cache handler");
    caches.lemma_cache.invalidate_all();
    caches.embeddings_cache.invalidate_all();
    Ok(std::borrow::Cow::Borrowed("cache cleaned"))
}
