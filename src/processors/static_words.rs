use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Ok;
use async_trait::async_trait;

use crate::handlers::data::{Processor, WorkContext, WorkMI};
use crate::utils::perf::PerfLogger;
use crate::utils::strings::is_number;

pub struct StaticWords {
    vocab: HashMap<String, String>,
}

impl StaticWords {
    pub fn new() -> anyhow::Result<StaticWords> {
        let _perf_log = PerfLogger::new("init static mi map");
        let vocab = init_vocab()?;
        log::info!("loaded clitics len {}", vocab.len());
        let res = StaticWords { vocab };
        Ok(res)
    }

    fn try_find(&self, w: &str) -> Option<String> {
        if is_number(w) {
            return Some("M----d-".to_string());
        }
        if let Some(s) = self.vocab.get(w) {
            return Some(s.clone());
        }
        if starts_with_nonalpha_num(w) {
            return Some("X-".to_string());
        }
        None
    }
}

fn starts_with_nonalpha_num(w: &str) -> bool {
    let c = w.chars().next().unwrap_or('-');
    !c.is_alphanumeric()
}

#[async_trait]
impl Processor for StaticWords {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("statics mapper");
        for sent in ctx.sentences.iter_mut() {
            for word_info in sent.iter_mut() {
                if word_info.is_word && word_info.mis.is_none() {
                    if let Some(res) = self.try_find(&word_info.w.to_lowercase()) {
                        word_info.mis = Some(Arc::new(vec![WorkMI {
                            lemma: None,
                            mi: Some(res),
                        }]));
                    }
                }
            }
        }
        Ok(())
    }
}

fn to_e(k: &str, v: &str) -> (String, String) {
    (k.to_string(), v.to_string())
}

fn init_vocab() -> anyhow::Result<HashMap<String, String>> {
    let mut res_map: HashMap<String, String> = HashMap::from([
        to_e(".", "Tp"),
        to_e(",", "Tc"),
        to_e(";", "Ts"),
        to_e(":", "Tn"),
        to_e("?", "Tq"),
        to_e("?..", "Tq"),
        to_e("!", "Te"),
        to_e("...", "Ti"),
        to_e("…", "Ti"),
        to_e("-", "Th"),
        to_e("–", "Th"),
        to_e("—", "Th"),
        to_e("(", "Tl"),
        to_e("[", "Tl"),
        to_e("{", "Tl"),
        to_e(")", "Tr"),
        to_e("]", "Tr"),
        to_e("}", "Tr"),
        to_e("/", "Tt"),
        to_e("'", "Tu"),
        to_e("\"", "Tu"),
        to_e("„", "Tu"),
        to_e("“", "Tu"),
        to_e("‘", "Tu"),
        to_e("\"", "Tu"),
    ]);
    for o in ["|", "\\", "*", "%", "^", "$", "•", "+"] {
        res_map.insert(o.to_string(), "Tx".to_string());
    }
    Ok(res_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn test_try_find(input: &str, expected: Option<String>) {
        let p = StaticWords::new().unwrap();
        let res = p.try_find(input);
        assert_eq!(res, expected);
    }

    macro_rules! try_find_test {
        ($suite:ident, $($name:ident: $input:expr, $expected:expr,)*) => {
            mod $suite {
                use super::*;
                $(
                    #[test]
                    fn $name() {
                        test_try_find($input, $expected);
                    }
                )*
            }
        }
    }

    try_find_test!(try_find_scope,
        number: "10", Some("M----d-".to_string()),
        coma: ",", Some("Tc".to_string()),
        dot: ".", Some("Tp".to_string()),
        some: "=", Some("X-".to_string()),
        with_dor: ".olia", Some("X-".to_string()),
    );
}
