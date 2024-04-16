use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};

use anyhow::Ok;
use async_trait::async_trait;

use crate::handlers::data::{Processor, WorkContext};
use crate::utils::perf::PerfLogger;

pub struct TagsMapper {
    vocab: HashMap<i32, String>,
}

impl TagsMapper {
    pub fn new(file_str: &str) -> anyhow::Result<TagsMapper> {
        let _perf_log = PerfLogger::new("tags loader");
        log::info!("Loading vocab from {}", file_str);
        let tags = read_vocab_file(file_str)?;
        let mut vocab_lookup: HashMap<i32, String> = HashMap::new();
        for (index, tag) in tags.into_iter().enumerate() {
            vocab_lookup.insert(index as i32, tag);
        }
        log::info!("loaded vocab len {}", vocab_lookup.len());

        let res = TagsMapper {
            vocab: vocab_lookup,
        };
        Ok(res)
    }
}

#[async_trait]
impl Processor for TagsMapper {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("tags mapper");
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                if let Some(id) = word_info.predicted {
                    match self.vocab.get(&id) {
                        Some(tag) => word_info.predicted_str = Some(tag.to_string()),
                        None => log::warn!("No tag by id {id}"),
                    };
                }
            }
        }
        Ok(())
    }
}

fn read_vocab_file(filename: &str) -> anyhow::Result<Vec<String>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut vocab = Vec::new();
    for line in reader.lines() {
        let word = line?;
        vocab.push(word.trim().to_string());
    }

    Ok(vocab)
}
