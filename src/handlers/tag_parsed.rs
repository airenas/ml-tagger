use std::sync::Arc;

use crate::{
    handlers::{
        data::{Service, WorkContext, WorkWord},
        tag::{map_res, process_line},
    },
    utils::perf::PerfLogger,
};
use anyhow::Context;
use tokio::sync::RwLock;

use super::data::{ApiResult, TagParams, Word};

use axum::{
    debug_handler,
    extract::{self, Query, State},
    Json,
};

#[debug_handler]
pub async fn handler(
    State(srv_wrap): State<Arc<RwLock<Service>>>,
    Query(params): Query<TagParams>,
    Json(input): Json<Vec<Vec<String>>>,
) -> ApiResult<extract::Json<Vec<Word>>> {
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

    process_line(&srv, &mut ctx).await.context("process")?;
    let res = map_res(ctx).context("map res")?;
    Ok(Json(res))
}
