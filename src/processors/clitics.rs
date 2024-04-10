use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};

use anyhow::Ok;
use async_trait::async_trait;

use crate::handlers::data::{Processor, WorkContext, WorkMI};
use crate::utils::PerfLogger;

pub struct Clitics {
    vocab: HashMap<String, Vec<String>>,
}

impl Clitics {
    pub fn new(file_str: &str) -> anyhow::Result<Clitics> {
        let _perf_log = PerfLogger::new("clitics loader");
        log::info!("Loading clitics from {}", file_str);
        let vocab = read_clitics_file(file_str)?;
        log::info!("loaded clitics len {}", vocab.len());
        let res = Clitics { vocab };
        Ok(res)
    }
}

#[async_trait]
impl Processor for Clitics {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("tags mapper");
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                if let Some(res) = self.vocab.get(&word_info.w.to_lowercase()) {
                    word_info.mis = Some(
                        res.into_iter()
                            .map(|f| WorkMI {
                                mi: Some(f.clone()),
                                lemma: None,
                            })
                            .collect(),
                    );
                }
            }
        }
        Ok(())
    }
}

fn read_clitics_file(filename: &str) -> anyhow::Result<HashMap<String, Vec<String>>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut clitics_map: HashMap<String, Vec<String>> = HashMap::new();
    for line in reader.lines() {
        let (key, value) = parse_line(line?)?;
        clitics_map.insert(key, value);
    }

    Ok(clitics_map)
}

fn parse_line(line: String) -> anyhow::Result<(String, Vec<String>)> {
    let parts: Vec<&str> = line.split('\t').collect();
    if let Some(p) = parts.get(1) {
        let mis: Vec<String> = p.split(';').map(|f| f.split(":").collect()).collect();
        let res: Vec<String> = mis.into_iter().filter(|f| f.len() > 0).collect();
        if res.len() == 0 {
            return Err(anyhow::anyhow!("failed parse line: {}", line));
        }
        return Ok((parts.get(0).unwrap().to_string(), res));
    }
    return Err(anyhow::anyhow!("failed parse line: {}", line));
}
