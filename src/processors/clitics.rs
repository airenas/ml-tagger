use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::sync::Arc;

use anyhow::Ok;
use async_trait::async_trait;

use crate::handlers::data::{Processor, WorkContext, WorkMI};
use crate::utils::perf::PerfLogger;

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
        let _perf_log = PerfLogger::new("clitics mapper");
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                if word_info.is_word {
                    if let Some(res) = self.vocab.get(&word_info.w.to_lowercase()) {
                        word_info.mis = Some(Arc::new(
                            res.iter()
                                .map(|f| WorkMI {
                                    mi: Some(f.clone()),
                                    lemma: None,
                                })
                                .collect(),
                        ));
                    }
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
        let mis = p
            .split(';')
            .map(|f| f.split(':').skip(1).take(1).collect())
            .collect::<Vec<String>>();
        let res: Vec<String> = mis.into_iter().filter(|f| !f.is_empty()).collect();
        if res.is_empty() {
            return Err(anyhow::anyhow!("failed parse line: {}", line));
        }
        let w = parts.first().unwrap().trim().to_string();
        if w.is_empty() {
            return Err(anyhow::anyhow!("failed parse line: {}: no word", line));
        }
        return Ok((w, res));
    }
    Err(anyhow::anyhow!("failed parse line: {}", line))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn test_parse(input: String, expected: (String, Vec<String>)) {
        assert_eq!(parse_line(input).unwrap(), expected);
    }

    macro_rules! parse_line_test {
        ($suite:ident, $($name:ident: $input:expr, $expected:expr,)*) => {
            mod $suite {
                use super::*;
                $(
                    #[test]
                    fn $name() {
                        test_parse($input, $expected);
                    }
                )*
            }
        }
    }

    parse_line_test!(parse_line_ok,
        one_mi: "olia ol\t:aaa".to_string(), ("olia ol".to_string(), vec!["aaa".to_string()]),
        two_mi: "olia ol\taaa:Pgfsdn;aaa:Pgmsdn".to_string(), ("olia ol".to_string(), vec!["Pgfsdn".to_string(), "Pgmsdn".to_string()]),
        more_mi: "olia more \t:11;:22;:33;:44".to_string(), ("olia more".to_string(), vec!["11".to_string(), "22".to_string(), "33".to_string(), "44".to_string()]),
    );

    fn test_parse_fail(input: String) {
        let res = parse_line(input);
        assert!(res.is_err());
    }

    macro_rules! parse_line_err_test {
        ($suite:ident, $($name:ident: $input:expr,)*) => {
            mod $suite {
                use super::*;
                $(
                    #[test]
                    fn $name() {
                        test_parse_fail($input);
                    }
                )*
            }
        }
    }

    parse_line_err_test!(parse_line_fail,
        no_line: "".to_string(),
        no_word: "\t:olia".to_string(),
        no_mi: "olia\t".to_string(),
        no_tab: "olia mi".to_string(),
    );
}
