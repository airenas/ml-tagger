use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};

use anyhow::Ok;
use async_trait::async_trait;

use crate::handlers::data::{Processor, WorkContext, WorkWord};
use crate::utils::PerfLogger;

pub struct Restorer {
    frequency_vocab: HashMap<String, Vec<(String, u32)>>,
}

impl Restorer {
    pub fn new(file_str: &str) -> anyhow::Result<Restorer> {
        let _perf_log = PerfLogger::new("restorer frequence loader");
        log::info!("Loading frequences from {}", file_str);
        let frequency_vocab = read_freq_file(file_str)?;
        log::info!("loaded frequences {}", frequency_vocab.len());
        let res = Restorer { frequency_vocab };
        Ok(res)
    }

    fn restore(
        &self,
        word_info: &WorkWord,
        freq_data: Option<&Vec<(String, u32)>>,
    ) -> anyhow::Result<(Option<String>, Option<String>)> {
        if let Some(mis) = word_info.mis.as_ref() {
            if mis.len() == 0 {
                return Ok((None, None));
            }
            if mis.len() == 1 {
                let mi = mis.get(0).unwrap();
                return Ok((mi.mi.clone(), mi.lemma.clone()));
            }
        }
        return Ok((None, None));
    }
}

#[async_trait]
impl Processor for Restorer {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("restorer mapper");
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                let freq_data = self.frequency_vocab.get(&word_info.w);
                let (mi, lemma) = self.restore(word_info, freq_data)?;
                word_info.mi = mi;
                word_info.lemma = lemma;
            }
        }
        Ok(())
    }
}

fn read_freq_file(filename: &str) -> anyhow::Result<HashMap<String, Vec<(String, u32)>>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut clitics_map: HashMap<String, Vec<(String, u32)>> = HashMap::new();
    for line in reader.lines() {
        let (key, value) = parse_line(line?)?;
        clitics_map.insert(key, value);
    }

    Ok(clitics_map)
}

fn parse_line(line: String) -> anyhow::Result<(String, Vec<(String, u32)>)> {
    let parts: Vec<&str> = line.split('\t').collect();
    if let Some(p) = parts.get(1) {
        let mi_parts: Vec<&str> = p.split_whitespace().collect();
        let res: anyhow::Result<Vec<(String, u32)>> = mi_parts
            .iter()
            .map(|&part| {
                let mut split = part.split(':');
                let mi = split
                    .next()
                    .ok_or(anyhow::anyhow!("no mi in {part}"))?
                    .trim()
                    .to_string();
                let freq = split
                    .next()
                    .ok_or(anyhow::anyhow!("no freq in {part}"))
                    .and_then(|arg| {
                        arg.trim()
                            .parse::<u32>()
                            .map_err(|err| anyhow::anyhow!("no freq in {part}: {err}"))
                    })?;
                Ok((mi.to_string(), freq))
            })
            .collect();
        let res: Vec<(String, u32)> = res?;
        if res.len() == 0 {
            return Err(anyhow::anyhow!("failed parse line: {}", line));
        }
        return Ok((parts.get(0).unwrap().to_string(), res));
    }
    return Err(anyhow::anyhow!("failed parse line: {}", line));
}
