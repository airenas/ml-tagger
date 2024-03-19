use std::sync::Arc;

use crate::handlers::data::{Result, Service, Word};
use tokio::sync::RwLock;
use warp::reply::Reply;

pub struct WorkWord {
    pub w: String,
    pub mi: Option<String>,
    pub lemma: Option<String>,
    pub w_type: Option<String>,
}

pub async fn handler(
    input: Vec<Vec<String>>,
    srv_wrap: Arc<RwLock<Service>>,
) -> Result<impl Reply> {
    log::debug!("tag handler");
    let mut workData = Vec::<Vec<WorkWord>>::new();
    let mut cw = 0;
    for sentence in input {
        let mut work_sentence = Vec::<WorkWord>::new();
        for word in sentence {
            work_sentence.push(WorkWord {
                w: word,
                mi: None,
                lemma: None,
                w_type: None,
            });
            cw += 1;
        }
        workData.push(work_sentence);
    }
    log::debug!("got {} sent, {} words", workData.len(), cw);
    let srv = srv_wrap.read().await;

    

    let mut res = Vec::<Vec<Word>>::new();
    for sentence in workData {
        let mut res_sentence = Vec::<Word>::new();
        for word in sentence {
            res_sentence.push(Word {
                w: word.w,
                mi: String::from(""),
                lemma: String::from(""),
                w_type: String::from(""),
            });
            cw += 1;
        }
        res.push(res_sentence);
    }
    Ok(warp::reply::json(&res).into_response())
}
