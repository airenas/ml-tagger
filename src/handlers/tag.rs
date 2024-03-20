use std::sync::Arc;

use crate::handlers::{
    data::{Result, Service, Word, WorkContext, WorkWord},
    errors::OtherError,
};
use tokio::sync::RwLock;
use warp::reply::Reply;

use super::data::TagParams;

pub async fn handler(
    params: TagParams,
    input: Vec<Vec<String>>,
    srv_wrap: Arc<RwLock<Service>>,
) -> Result<impl Reply> {
    log::debug!("tag handler");
    let mut ctx = WorkContext::new(params);
    let mut cw = 0;
    for sentence in input {
        let mut work_sentence = Vec::<WorkWord>::new();
        for w in sentence {
            work_sentence.push(WorkWord::new(w));
            cw += 1;
        }
        ctx.sentences.push(work_sentence);
    }
    log::debug!("got {} sent, {} words", ctx.sentences.len(), cw);
    let srv = srv_wrap.read().await;
    srv.embedder
        .process(&mut ctx)
        .map_err(|e| OtherError { msg: e.to_string() })?;
    srv.onnx
        .process(&mut ctx)
        .map_err(|e| OtherError { msg: e.to_string() })?;

    let mut res = Vec::<Vec<Word>>::new();
    for sentence in ctx.sentences {
        let mut res_sentence = Vec::<Word>::new();
        for word in sentence {
            res_sentence.push(Word {
                w: word.w,
                mi: String::from(""),
                lemma: String::from(""),
                w_type: String::from(""),
                embeddings: match ctx.params.debug {
                    Some(true) => word.embeddings,
                    _ => None,
                },
                predicted: match ctx.params.debug {
                    Some(true) => word.predicted,
                    _ => None,
                },
            });
            cw += 1;
        }
        res.push(res_sentence);
    }
    Ok(warp::reply::json(&res).into_response())
}
