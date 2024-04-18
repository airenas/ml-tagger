use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};

use anyhow::Ok;
use async_trait::async_trait;

use crate::handlers::data::{Processor, WorkContext, WorkMI, WorkWord};
use crate::utils::perf::PerfLogger;
use crate::utils::strings::is_number;

pub struct Restorer {
    frequency_vocab: HashMap<String, HashMap<String, u32>>,
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
        freq_data: Option<&HashMap<String, u32>>,
    ) -> anyhow::Result<(Option<String>, Option<String>)> {
        if let Some(predicted) = &word_info.predicted_str {
            if let Some(mis) = word_info.mis.as_ref() {
                if mis.len() == 1 {
                    let mi = mis.first().unwrap();
                    return Ok((mi.mi.clone(), mi.lemma.clone()));
                }
                if mis.len() > 1 {
                    let mi = restore(mis, predicted, freq_data);
                    return Ok((mi.mi, mi.lemma));
                }
            }
            Ok((None, Some(predicted.clone())))
        } else {
            Err(anyhow::anyhow!("no prediction for: {}", word_info.w))
        }
    }
}

#[async_trait]
impl Processor for Restorer {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("restorer mapper");
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                if word_info.is_word {
                    let freq_data = self.frequency_vocab.get(&word_info.w);
                    let (mi, lemma) = self.restore(word_info, freq_data)?;
                    word_info.mi = mi;
                    word_info.lemma = lemma;
                }
                word_info.w_type = get_type(word_info);
            }
        }
        Ok(())
    }
}

fn get_type(word_info: &WorkWord) -> Option<String> {
    if !word_info.is_word {
        return Some("SPACE".to_string());
    }
    let w = word_info.w.as_str();
    let mi = word_info.mi.as_deref().unwrap_or_default();
    if is_number(w) {
        return Some("NUMBER".to_string());
    } else if is_sep(mi) {
        return Some("SEPARATOR".to_string());
    }
    Some("WORD".to_string())
}

fn is_sep(mi: &str) -> bool {
    mi.starts_with('T')
}

fn read_freq_file(filename: &str) -> anyhow::Result<HashMap<String, HashMap<String, u32>>> {
    let file = File::open(filename)?;
    let reader = io::BufReader::new(file);
    let mut clitics_map: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for line in reader.lines() {
        let (key, value) = parse_line(line?)?;
        clitics_map.insert(key, value);
    }

    Ok(clitics_map)
}

fn parse_line(line: String) -> anyhow::Result<(String, HashMap<String, u32>)> {
    let parts: Vec<&str> = line.split('\t').collect();
    if let Some(p) = parts.get(1) {
        let mi_parts: Vec<&str> = p.split_whitespace().collect();
        let res: anyhow::Result<HashMap<String, u32>> = mi_parts
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
                if mi.is_empty() {
                    return Err(anyhow::anyhow!("failed parse line: {}: no mi", line));
                }
                Ok((mi, freq))
            })
            .collect();
        let res: HashMap<String, u32> = res?;
        let w = parts.first().unwrap().trim().to_string();
        if w.is_empty() {
            return Err(anyhow::anyhow!("failed parse line: {}: no word", line));
        }
        if res.is_empty() {
            return Err(anyhow::anyhow!("failed parse line: {}", line));
        }
        return Ok((w, res));
    }
    Err(anyhow::anyhow!("failed parse line: {}", line))
}

fn half_change(pos: char, predited: char, t: char, i: usize) -> bool {
    match pos {
        'N' => {
            if (i == 2 && predited == 'f' && t == 'c') || (i == 3 && predited == 'p' && t == 'd') {
                return true;
            }
        }
        'A' => {
            if (i == 3 && predited == 'f' && t == 'n') || (i == 4 && predited == 'p' && t == 'd') {
                return true;
            }
        }
        'P' => {
            if i == 3 && predited == 'p' && t == 'd' {
                return true;
            }
        }
        _ => {}
    }
    false
}

fn calc(p: &str, t: &str) -> f64 {
    if p.chars().next() != t.chars().next() {
        return 50.0;
    }
    let p_first = p.chars().next().unwrap_or('-');
    let mut res = 0.0;
    for (i, (pv, tv)) in p.chars().zip(t.chars()).enumerate() {
        if pv != tv {
            if half_change(p_first, pv, tv, i) {
                res += 0.03;
            } else if pv != '-' {
                res += 1.0;
            } else {
                res += 0.01;
            }
        }
    }
    res
}

fn restore(all: &Vec<WorkMI>, predicted: &str, freq_data: Option<&HashMap<String, u32>>) -> WorkMI {
    let mut bv = 1000.0;
    let empty_res = WorkMI {
        mi: None,
        lemma: None,
    };
    let mut res: &WorkMI = &empty_res;
    for t in all {
        let mut v = calc(predicted, t.mi.as_ref().map_or("", |f| f.as_str()));
        let freq_p = 0.001 / (get_freq(freq_data, &t.mi) + 1.0);
        v += freq_p;
        if v < bv {
            bv = v;
            res = t;
        }
    }
    res.clone()
}

fn get_freq(freq_data: Option<&HashMap<String, u32>>, mi: &Option<String>) -> f64 {
    if let Some(fd) = freq_data {
        if let Some(mi_v) = mi {
            let v = fd.get(mi_v);
            return v.map_or(0.0, |f| *f as f64);
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn test_parse(input: String, expected: (String, HashMap<String, u32>)) {
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
        one_mi: "olia ol\taaa:10".to_string(), ("olia ol".to_string(), HashMap::from([("aaa".to_string(), 10)])),
        two_mi: "olia ol\tPgfsdn:2 Pgmsdn:1".to_string(), ("olia ol".to_string(), HashMap::from([("Pgfsdn".to_string(), 2), ("Pgmsdn".to_string(), 1)])),
        more_mi: "olia \taa:11 bb:22 cc:33 dd:44".to_string(), ("olia".to_string(), HashMap::from([("aa".to_string(), 11), ("bb".to_string(), 22), ("cc".to_string(), 33),  ("dd".to_string(), 44)])),
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
        no_word: "\tolia:10".to_string(),
        bad_number: "aaa\tolia:b10".to_string(),
        no_mi: "aukštą\t:15 Agpfsan:14 Ncmsan-:13".to_string(),
        no_tab: "aukštą aaa:15 Agpfsan:14 Ncmsan-:13".to_string(),
    );

    fn test_half_change(pos: char, predicted: char, t: char, i: usize, expected: bool) {
        let got = half_change(pos, predicted, t, i);
        assert_eq!(got, expected);
    }

    macro_rules! half_change_test {
        ($suite:ident, $($name:ident: $pos:expr, $predicted:expr, $t:expr, $i:expr, $expected:expr,)*) => {
            mod $suite {
                use super::*;
                $(
                    #[test]
                    fn $name() {
                        test_half_change($pos, $predicted, $t, $i, $expected);
                    }
                )*
            }
        }
    }

    half_change_test!(half_change_test,
        dktv: 'N', 'f', 'c', 2, true,
        bdv: 'A', 'f', 'n', 3, true,
        false_bdv: 'A', 'f', 'n', 2, false,
    );
}
