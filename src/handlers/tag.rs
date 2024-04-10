use std::sync::Arc;

use crate::{
    handlers::{
        data::{Result, Service, Word, WorkContext, WorkWord, MI},
        errors::OtherError,
    },
    utils::PerfLogger,
};
use tokio::sync::RwLock;
use warp::reply::Reply;

use super::data::TagParams;

pub async fn handler(
    params: TagParams,
    input: Vec<Vec<String>>,
    srv_wrap: Arc<RwLock<Service>>,
) -> Result<impl Reply> {
    let _perf_log = PerfLogger::new("tag handler");
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
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;
    srv.onnx
        .process(&mut ctx)
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;
    srv.tag_mapper
        .process(&mut ctx)
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;
    srv.clitics
        .process(&mut ctx)
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;
    srv.lemmatize_words_mapper
        .process(&mut ctx)
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;

    let mut res = Vec::<Vec<Word>>::new();
    for sentence in ctx.sentences {
        let mut res_sentence = Vec::<Word>::new();
        for word in sentence {
            res_sentence.push(Word {
                w: word.w,
                mi: None,
                lemma: None,
                w_type: None,
                embeddings: match is_wanted(&ctx.params.debug, "emb:") {
                    Some(true) => word.embeddings,
                    _ => None,
                },
                predicted: match is_wanted(&ctx.params.debug, "predicted:") {
                    Some(true) => word.predicted,
                    _ => None,
                },
                predicted_str: match is_wanted(&ctx.params.debug, "predicted_str") {
                    Some(true) => word.predicted_str,
                    _ => None,
                },
                mis: match is_wanted(&ctx.params.debug, "mis") {
                    Some(true) => match word.mis {
                        Some(mis) => {
                            let mis_res = mis
                                .iter()
                                .map(|mi| MI {
                                    lemma: mi.lemma.clone(),
                                    mi: mi.mi.clone(),
                                })
                                .collect();
                            Some(mis_res)
                        }
                        _ => None,
                    },
                    _ => None,
                },
            });
            cw += 1;
        }
        res.push(res_sentence);
    }

    Ok(warp::reply::json(&res).into_response())
}

fn is_wanted(debug: &Option<String>, arg: &str) -> Option<bool> {
    match debug {
        Some(v) => {
            if v.contains(arg) {
                return Some(true);
            }
            return None;
        }
        None => None,
    }
}
