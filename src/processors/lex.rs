use std::collections::HashSet;
use std::time::Duration;

use anyhow::Ok;
use async_trait::async_trait;
use reqwest::Client;
use reqwest_retry::policies::ExponentialBackoff;
use reqwest_retry::RetryTransientMiddleware;
use serde::Deserialize;

use crate::handlers::data::{Processor, WorkContext, WorkWord};
use crate::utils::perf::PerfLogger;
use crate::utils::strings;
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use tokio::sync::Mutex;

pub struct Lexer {
    client: Mutex<ClientWithMiddleware>,
    url: String,
    additional_split: HashSet<char>,
}

#[derive(Debug, Deserialize)]
struct LexResponse {
    seg: Vec<Vec<i32>>,
    s: Vec<Vec<i32>>,
    // p: Vec<Vec<i32>>,
}

const ADDITIONAL_SPLITTERS: &str = "-‘\"–‑/:;`−≤≥⁰'";

impl Lexer {
    pub fn new(url_str: &str) -> anyhow::Result<Lexer> {
        log::info!("lex url: {url_str}");
        let additional_split = HashSet::from_iter(ADDITIONAL_SPLITTERS.chars());

        let retry_policy = ExponentialBackoff::builder().build_with_max_retries(3);
        let client = Client::builder().timeout(Duration::from_secs(10)).build()?;
        let client_with_retry = ClientBuilder::new(client)
            .with(RetryTransientMiddleware::new_with_policy(retry_policy))
            .build();

        let locked_client: Mutex<ClientWithMiddleware> = Mutex::new(client_with_retry);
        let res = Lexer {
            client: locked_client,
            url: url_str.to_string(),
            additional_split,
        };
        Ok(res)
    }

    async fn split(&self, text: &str) -> anyhow::Result<Vec<Vec<(String, bool)>>> {
        let _perf_log = PerfLogger::new("call lex");
        log::info!("lex - text len:'{}'", text.len());
        let client = self.client.lock().await;
        let _perf_real_log = PerfLogger::new("real lex");
        let response = client
            .post(self.url.clone())
            .header("Content-Type", "application/json")
            .body(text.to_string())
            .send()
            .await
            .map_err(|err| anyhow::anyhow!("failed to send request: {}", err))?;

        log::info!("resp'{}'", response.status().as_u16());
        if response.status().is_success() {
            let resp_res: LexResponse = response
                .json()
                .await
                .map_err(|err| anyhow::anyhow!("failed to deserialize response: {}", err))?;
            let res = self.convert(resp_res, text)?;
            return Ok(res);
        };
        let status = response.status().as_u16();
        let body = response.bytes().await?;
        let body_str = String::from_utf8_lossy(&body);
        Err(anyhow::anyhow!(
            "Failed to make request: {} {}",
            status,
            body_str
        ))?
    }

    fn convert(
        &self,
        resp_res: LexResponse,
        text: &str,
    ) -> anyhow::Result<Vec<Vec<(String, bool)>>> {
        let seg: Vec<(i32, i32, i32)> = group_sentences(resp_res)?;
        let mut res: Vec<Vec<(String, bool)>> = Vec::new();
        let mut current: Vec<(String, bool)> = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut last_seen = 0;
        let mut last_index = 0;
        for v in seg.iter() {
            if v.2 != last_seen {
                if !current.is_empty() {
                    res.push(current);
                    current = Vec::new();
                }
                last_seen = v.2;
            }
            if last_index < v.0 {
                let chars: Vec<char> = chars[last_index as usize..v.0 as usize].to_vec();
                let s: String = chars.iter().collect();
                current.push((s, false)); // spaces
            }
            let words = get_string(&chars, v.0 as usize, v.1 as usize, &self.additional_split);
            for w in words {
                current.push((w, true)); // words
            }
            last_index = v.1
        }
        if !current.is_empty() {
            res.push(current);
        }
        Ok(res)
    }
}

fn get_string(
    text: &[char],
    v_1: usize,
    v_2: usize,
    additional_split: &HashSet<char>,
) -> Vec<String> {
    let chars: Vec<char> = text[v_1..v_2].to_vec();
    let s: String = chars.iter().collect();
    if chars.len() == 1 {
        return vec![s];
    }
    if strings::is_number(s.as_str()) {
        return vec![s];
    }
    if strings::is_url(s.as_str()) {
        return vec![s];
    }
    if strings::is_email(s.as_str()) {
        return vec![s];
    }
    try_split(&chars, additional_split)
}

fn try_split(chars: &[char], additional_split: &HashSet<char>) -> Vec<String> {
    let mut res = Vec::<String>::new();
    let mut last = 0;
    for (index, ch) in chars.iter().enumerate() {
        if additional_split.contains(ch) {
            if last != index {
                res.push(chars[last..index].iter().collect());
            }
            res.push(ch.to_string());
            last = index + 1;
        }
    }
    if last < chars.len() {
        res.push(chars[last..].iter().collect());
    }
    res
}

fn group_sentences(resp_res: LexResponse) -> anyhow::Result<Vec<(i32, i32, i32)>> {
    let mut res = Vec::<(i32, i32, i32)>::new();
    for seg in resp_res.seg {
        if seg.len() != 2 {
            return Err(anyhow::anyhow!("wrong seg, len != 2"));
        }
        res.push((seg[0], seg[0] + seg[1], 0));
    }
    let mut s_i: usize = 0;
    for v in res.iter_mut() {
        v.2 = s_i as i32;
        if resp_res.s.len() < s_i {
            return Err(anyhow::anyhow!("wrong s, len < {}", s_i));
        }
        let sv: &Vec<i32> = &resp_res.s[s_i];
        if sv.len() != 2 {
            return Err(anyhow::anyhow!("wrong s, len != 2"));
        }
        if v.1 > sv[0] + sv[1] {
            s_i += 1;
            v.2 = s_i as i32;
        }
    }
    Ok(res)
}

#[async_trait]
impl Processor for Lexer {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("lex text");
        let sentences = self
            .split(&ctx.text)
            .await
            .map_err(|err| anyhow::anyhow!("lex failure: {}", err))?;
        for sentence in sentences {
            let mut work_sentence = Vec::<WorkWord>::new();
            for (w, is_word) in sentence {
                work_sentence.push(WorkWord::new(w, is_word));
            }
            ctx.sentences.push(work_sentence);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn test_try_split(input: &str, expected: Vec<String>) {
        let chars: Vec<char> = input.chars().collect();
        let additional_split = HashSet::from_iter(ADDITIONAL_SPLITTERS.chars());
        assert_eq!(try_split(&chars, &additional_split), expected);
    }

    macro_rules! try_split_test {
        ($suite:ident, $($name:ident: $input:expr, $expected:expr,)*) => {
            mod $suite {
                use super::*;
                $(
                    #[test]
                    fn $name() {
                        test_try_split($input, $expected);
                    }
                )*
            }
        }
    }

    try_split_test!(split_word,
        no_split: "olia", vec!["olia".to_string()],
        split1: "olia-olia", vec!["olia".to_string(), "-".to_string(), "olia".to_string()],
        split_several: "olia-olia-2", vec!["olia".to_string(), "-".to_string(), "olia".to_string(),"-".to_string(), "2".to_string()],
        split_lt: "dvidešimtį-dvi", vec!["dvidešimtį".to_string(), "-".to_string(), "dvi".to_string()],
        split_other: "dvidešimtį≤≥⁰dvi", vec!["dvidešimtį".to_string(), "≤".to_string(),"≥".to_string(),"⁰".to_string(), "dvi".to_string()],
        split_last: "dvidešimtį-", vec!["dvidešimtį".to_string(), "-".to_string()],
        split_first: "-dvi", vec!["-".to_string(), "dvi".to_string()],
    );
}
