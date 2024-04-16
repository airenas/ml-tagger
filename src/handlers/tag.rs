use std::sync::Arc;

use crate::{
    handlers::{
        data::{Result, Service, Word, WorkContext, MI},
        errors::{OtherError, ParamError},
    },
    utils::perf::PerfLogger,
};
use tokio::sync::RwLock;
use tokio_util::bytes::Bytes;
use warp::reply::Reply;

use super::data::TagParams;

pub async fn handler(
    params: TagParams,
    input: Bytes,
    srv_wrap: Arc<RwLock<Service>>,
) -> Result<impl Reply> {
    let _perf_log = PerfLogger::new("tag handler");

    let s = match std::str::from_utf8(&input) {
        Ok(v) => Ok(v),
        Err(_) => Err(ParamError {
            msg: "no utf-8 input".to_string(),
        }),
    }?;

    let mut ctx = WorkContext::new(params, s.trim().to_string());

    let srv = srv_wrap.read().await;
    srv.lexer
        .process(&mut ctx)
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;

    let cw: usize = ctx.sentences.iter().map(|f| f.len()).sum();
    log::debug!("got {} sent, {} words", ctx.sentences.len(), cw);

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
    srv.static_words
        .process(&mut ctx)
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;
    srv.lemmatize_words_mapper
        .process(&mut ctx)
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;
    srv.restorer
        .process(&mut ctx)
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;

    let mut res = Vec::<Vec<Word>>::new();
    for sentence in ctx.sentences {
        let mut res_sentence = Vec::<Word>::new();
        for word in sentence {
            res_sentence.push(Word {
                w: word.w,
                mi: word.mi,
                lemma: word.lemma,
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
            None
        }
        None => None,
    }
}
