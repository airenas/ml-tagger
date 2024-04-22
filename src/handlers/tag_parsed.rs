use std::sync::Arc;

use crate::{
    handlers::{
        data::{Result, Service, WorkContext, WorkWord},
        errors::OtherError,
        tag::{map_res, process_line},
    },
    utils::perf::PerfLogger,
};
use tokio::sync::RwLock;
use warp::reply::Reply;

use super::data::TagParams;

pub async fn handler(
    params: TagParams,
    input: Vec<Vec<String>>,
    srv_wrap: Arc<RwLock<Service>>,
) -> Result<impl Reply> {
    let _perf_log = PerfLogger::new("tag parsed handler");
    let mut ctx = WorkContext::new(params, "".to_string());
    for sentence in input {
        let mut work_sentence = Vec::<WorkWord>::new();
        for w in sentence {
            work_sentence.push(WorkWord::new(w, true));
        }
        ctx.sentences.push(work_sentence);
    }
    let srv = srv_wrap.read().await;

    process_line(&srv, &mut ctx)
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;
    let res = map_res(ctx).map_err(|e| OtherError { msg: e.to_string() })?;
    Ok(warp::reply::json(&res).into_response())
}
