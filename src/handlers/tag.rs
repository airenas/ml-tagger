use std::sync::Arc;

use crate::{
    handlers::data::{Service, Word, WorkContext, MI},
    utils::perf::PerfLogger,
};
use anyhow::Context;
use axum::{
    debug_handler,
    extract::{self, Query, State},
    Json,
};
use tokio::sync::RwLock;

use super::data::{ApiResult, TagParams, WordType};

#[debug_handler]
pub async fn handler(
    State(srv_wrap): State<Arc<RwLock<Service>>>,
    Query(params): Query<TagParams>,
    string_body: String,
) -> ApiResult<extract::Json<Vec<Word>>> {
    let _perf_log = PerfLogger::new("tag handler");

    let mut ctx = WorkContext::new(params, string_body.trim().to_string());

    let srv = srv_wrap.read().await;
    srv.lexer.process(&mut ctx).await.context("lex")?;

    let cw: usize = ctx.sentences.iter().map(|f| f.len()).sum();
    log::debug!("got {} sent, {} words", ctx.sentences.len(), cw);

    process_line(&srv, &mut ctx).await.context("process")?;

    let res = map_res(ctx).context("map res")?;
    Ok(Json(res))
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
            w_type: WordType::SentenceEnd,
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
