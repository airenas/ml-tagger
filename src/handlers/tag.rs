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

    process_line(&srv, &mut ctx)
        .await
        .map_err(|e| OtherError { msg: e.to_string() })?;

    let res = map_res(ctx).map_err(|e| OtherError { msg: e.to_string() })?;
    Ok(warp::reply::json(&res).into_response())
}

pub async fn process_line(
    srv: &tokio::sync::RwLockReadGuard<'_, Service>,
    ctx: &mut WorkContext,
) -> anyhow::Result<()> {
    srv.embedder.process(ctx).await?;
    srv.onnx.process(ctx).await?;
    srv.tag_mapper.process(ctx).await?;
    srv.clitics.process(ctx).await?;
    srv.static_words.process(ctx).await?;
    srv.lemmatize_words_mapper.process(ctx).await?;
    srv.restorer.process(ctx).await?;
    Ok(())
}

pub fn map_res(ctx: WorkContext) -> anyhow::Result<Vec<Word>> {
    let mut res = Vec::<Word>::new();
    for sentence in ctx.sentences {
        for word in sentence {
            res.push(Word {
                w: Some(word.w),
                mi: word.mi,
                lemma: word.lemma,
                w_type: word.w_type,
                embeddings: match is_wanted(&ctx.params.debug, "emb:") {
                    Some(true) => word.embeddings.map(|arc| Arc::clone(&arc).to_vec()),
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
        res.push(Word {
            w_type: Some("SENTENCE_END".to_string()),
            w: None,
            mi: None,
            lemma: None,
            embeddings: None,
            predicted: None,
            predicted_str: None,
            mis: None,
        });
    }
    Ok(res)
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
