use std::collections::HashSet;
use std::time::Duration;

use anyhow::{Ok};
use async_trait::async_trait;
use serde::Deserialize;

use crate::handlers::data::{Processor, WorkContext, WorkWord};
use crate::utils::perf::PerfLogger;
use crate::utils::strings;
use reqwest::Client;

pub struct Lexer {
    client: Client,
    url: String,
    additional_split: HashSet<char>,
}

#[derive(Debug, Deserialize)]
struct LexResponse {
    seg: Vec<Vec<i32>>,
    s: Vec<Vec<i32>>,
    // p: Vec<Vec<i32>>,
}

impl Lexer {
    pub fn new(url_str: &str) -> anyhow::Result<Lexer> {
        log::info!("lex url: {url_str}");
        let additional_split = HashSet::from_iter("-‘\"–‑/:;`−≤≥⁰'".chars());
        let res = Lexer {
            client: Client::builder().timeout(Duration::from_secs(10)).build()?,
            url: url_str.to_string(),
            additional_split,
        };
        Ok(res)
    }

    async fn split(&self, text: &str) -> anyhow::Result<Vec<Vec<String>>> {
        let _perf_log = PerfLogger::new("call lex");
        log::info!("lex - text len:'{}'", text.len());
        let response = self
            .client
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
                .map_err(|err| anyhow::anyhow!("Failed to deserialize response: {}", err))?;
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

    fn convert(&self, resp_res: LexResponse, text: &str) -> anyhow::Result<Vec<Vec<String>>> {
        let seg: Vec<(i32, i32, i32)> = group_sentences(resp_res)?;
        let mut res: Vec<Vec<String>> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut last_seen = 0;
        for v in seg.iter() {
            if v.2 != last_seen {
                if !current.is_empty() {
                    res.push(current);
                    current = Vec::new();
                }
                last_seen = v.2;
            }
            current.extend(get_string(
                &chars,
                v.0 as usize,
                v.1 as usize,
                &self.additional_split,
            ))
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
    let mut res = Vec::<String>::new();
    let mut last = 0;
    for (index, matched) in s.match_indices(|c: char| additional_split.contains(&c)) {
        if last != index {
            res.push(chars[last..index].iter().collect());
        }
        res.push(matched.to_string());
        last = index + matched.len();
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
        let sentences = self.split(&ctx.text).await?;
        for sentence in sentences {
            let mut work_sentence = Vec::<WorkWord>::new();
            for w in sentence {
                work_sentence.push(WorkWord::new(w));
            }
            ctx.sentences.push(work_sentence);
        }

        Ok(())
    }
}
