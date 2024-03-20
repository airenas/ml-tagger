use std::sync::Arc;

use tokio::sync::RwLock;
use warp::Reply;

use super::data::{LiveResponse, Result, Service};

pub async fn handler(srv_wrap: Arc<RwLock<Service>>) -> Result<impl Reply> {
    log::debug!("live handler");
    let _srv = srv_wrap.read().await;
    let res = LiveResponse { status: true };
    Ok(warp::reply::json(&res).into_response())
}
