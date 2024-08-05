use std::sync::Arc;

use moka::future::Cache;
use warp::Reply;

use super::data::{Result, WorkMI};

pub async fn handler(
    lemma_cache: Cache<String, Arc<Vec<WorkMI>>>,
    embeddingd_cache: Cache<String, Arc<Vec<f32>>>,
) -> Result<impl Reply> {
    log::debug!("clean cache handler");
    lemma_cache.invalidate_all();
    embeddingd_cache.invalidate_all();
    Ok("cache cleaned")
}
