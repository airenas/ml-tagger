use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Error, Ok};
use async_trait::async_trait;
use serde::Deserialize;

use crate::handlers::data::{Processor, WorkContext, WorkMI};
use crate::utils::PerfLogger;
use reqwest::Client;

pub struct LemmatizeWordsMapper {
    client: Client,
    url: String,
}

#[derive(Debug, Deserialize)]
struct MI {
    mf: String,
    // mi: String,
    mi_vdu: String,
    // mih: String,
    // mis: String,
    // origin: String,
    // original: String,
}

#[derive(Debug, Deserialize)]
struct LemmaResponse {
    // ending: String,
    mi: Vec<MI>,
    // suffix: String,
    // word: String,
}

impl LemmatizeWordsMapper {
    pub fn new(url_str: &str) -> anyhow::Result<LemmatizeWordsMapper> {
        let res = LemmatizeWordsMapper {
            client: Client::builder().timeout(Duration::from_secs(10)).build()?,
            url: url_str.to_string(),
        };
        Ok(res)
    }

    async fn lemmatize(
        &self,
        map: &mut HashMap<String, Option<Vec<WorkMI>>>,
    ) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("lemmatizing");
        for (key, value) in map.iter_mut() {
            let new_value = self.make_request(key).await?;
            *value = new_value;
        }
        Ok(())
    }

    async fn make_request(&self, key: &str) -> anyhow::Result<Option<Vec<WorkMI>>> {
        let _perf_log = PerfLogger::new(format!("lemmatize '{}'", key).as_str());
        let url_str = format!("{}/{}?human=true&origin=true", self.url, key);
        log::info!("call: {url_str}");
        let response = self
            .client
            .get(url_str)
            .send()
            .await
            .map_err(|err| anyhow::anyhow!("failed to send request: {}", err))?;

        if response.status().is_success() {
            let resp_res: LemmaResponse = response
                .json()
                .await
                .map_err(|err| anyhow::anyhow!("Failed to deserialize response: {}", err))?;
            let res: Vec<WorkMI> = resp_res
                .mi
                .iter()
                .map(|mi| WorkMI {
                    lemma: Some(mi.mf.clone()),
                    mi: Some(mi.mi_vdu.clone()),
                })
                .collect();
            return Ok(Some(res));
        };
        let body = response.bytes().await?;
        let body_str = String::from_utf8_lossy(&body);
        Err(anyhow::anyhow!("Failed to make request: {}", body_str))?
    }
}

#[async_trait]
impl Processor for LemmatizeWordsMapper {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("lemmatize words");
        let mut words_map: HashMap<String, Option<Vec<WorkMI>>> = HashMap::new();

        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                words_map.insert(word_info.w.clone(), None);
            }
        }
        self.lemmatize(&mut words_map).await?;
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                match words_map.get(&word_info.w) {
                    Some(res) => {
                        if res.is_some() {
                            word_info.mis = res.clone().map(|vec| vec.clone());
                        } else {
                            Err(Error::msg(format!(
                                "word is not lemmatized {}",
                                &word_info.w
                            )))?
                        }
                    }
                    _ => Err(Error::msg(format!(
                        "word is not lemmatized {}",
                        &word_info.w
                    )))?,
                }
            }
        }

        Ok(())
    }
}
