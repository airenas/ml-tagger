use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Error, Ok};
use async_trait::async_trait;
use moka::future::Cache;
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::policies::ExponentialBackoff;
use reqwest_retry::RetryTransientMiddleware;
use serde::Deserialize;

use crate::handlers::data::{Processor, WorkContext, WorkMI};
use crate::utils::perf::PerfLogger;
use reqwest::Client;

pub struct LemmatizeWordsMapper {
    client: ClientWithMiddleware,
    url: String,
    cache: Cache<String, Arc<Vec<WorkMI>>>,
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
    pub fn new(
        url_str: &str,
        cache: Cache<String, Arc<Vec<WorkMI>>>,
    ) -> anyhow::Result<LemmatizeWordsMapper> {
        if !url_str.contains("{}") {
            return Err(anyhow::anyhow!(
                "lemma url {} does not contain {{}}",
                url_str
            ));
        }
        log::info!("lemma url: {url_str}");

        let retry_policy = ExponentialBackoff::builder().build_with_max_retries(3);
        let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
        let client_with_retry = ClientBuilder::new(client)
            .with(RetryTransientMiddleware::new_with_policy(retry_policy))
            .build();

        let res = LemmatizeWordsMapper {
            client: client_with_retry,
            url: url_str.to_string(),
            cache,
        };
        Ok(res)
    }

    async fn lemmatize(
        &self,
        map: &mut HashMap<String, Option<Vec<WorkMI>>>,
    ) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("lemmatizing");
        for (key, value) in map.iter_mut() {
            let cv = self.cache.get(key).await;
            if let Some(val) = cv {
                log::debug!("in lemma cache: {key}");
                let new_value = Arc::clone(&val);
                let extracted = (*new_value).clone();
                *value = Some(extracted);
            }
        }
        for (key, value) in map.iter_mut() {
            if value.is_none() {
                let new_value = self
                    .make_request(key)
                    .await
                    .map_err(|err| anyhow::anyhow!("lemma failure: {}", err))?;
                if let Some(val) = new_value {
                    *value = Some(val.clone());
                    self.cache.insert(key.clone(), Arc::new(val)).await;
                }
            }
        }
        Ok(())
    }

    async fn make_request(&self, key: &str) -> anyhow::Result<Option<Vec<WorkMI>>> {
        let _perf_log = PerfLogger::new(format!("lemmatize '{}'", key).as_str());
        let url_str = self.url.replace("{}", key);
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
            let mut res: Vec<WorkMI> = resp_res
                .mi
                .iter()
                .map(|mi| WorkMI {
                    lemma: Some(mi.mf.clone()),
                    mi: Some(mi.mi_vdu.clone()),
                })
                .collect();
            if res.is_empty() {
                res.push(WorkMI {
                    lemma: None,
                    mi: Some(fix_empty_lemma_res(key)),
                });
            }
            return Ok(Some(res));
        };
        let body = response.bytes().await?;
        let body_str = String::from_utf8_lossy(&body);
        if body_str.starts_with("N�ra �od�io") {
            log::warn!("failed to make request: {}", body_str);
            return Ok(Some(vec![WorkMI {
                lemma: None,
                mi: Some(fix_empty_lemma_res(key)),
            }]));
        }
        Err(anyhow::anyhow!("failed to make request: {}", body_str))?
    }
}

fn fix_empty_lemma_res(word: &str) -> String {
    if word.chars().count() == 1 {
        let c = word.chars().next().unwrap_or('-');
        if c.is_alphabetic() {
            return "Xr".to_string();
        }
    }
    "X-".to_string()
}

#[async_trait]
impl Processor for LemmatizeWordsMapper {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("lemmatize words");
        let mut words_map: HashMap<String, Option<Vec<WorkMI>>> = HashMap::new();

        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                if word_info.is_word && word_info.mis.is_none() {
                    words_map.insert(word_info.w.clone(), None);
                }
            }
        }
        self.lemmatize(&mut words_map).await?;
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                if word_info.is_word && word_info.mis.is_none() {
                    match words_map.get(&word_info.w) {
                        Some(res) => {
                            if res.is_some() {
                                word_info.mis = res.clone();
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
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_fix_empty_lemma_res() {
        assert_eq!(fix_empty_lemma_res("10"), "X-".to_string());
        assert_eq!(fix_empty_lemma_res("s"), "Xr".to_string());
        assert_eq!(fix_empty_lemma_res("ą"), "Xr".to_string(), "test ą");
        assert_eq!(fix_empty_lemma_res("š"), "Xr".to_string());
        assert_eq!(fix_empty_lemma_res("ss"), "X-".to_string());
    }
}
