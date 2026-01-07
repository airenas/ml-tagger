use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Ok;
use async_trait::async_trait;
use linkify::LinkFinder;

use crate::handlers::data::{Processor, WorkContext, WorkMI};
use crate::utils::perf::PerfLogger;
use crate::utils::strings::is_number;
use crate::{MATH_SYMBOLS, MI_EMAIL, MI_URL, SYMBOLS};

pub struct StaticWords {
    vocab: HashMap<String, Arc<Vec<WorkMI>>>,

    numeric_mi: Arc<Vec<WorkMI>>,
    x_mi: Arc<Vec<WorkMI>>,
    email_mi: Arc<Vec<WorkMI>>,
    url_mi: Arc<Vec<WorkMI>>,
}

impl StaticWords {
    pub fn new() -> anyhow::Result<StaticWords> {
        let _perf_log = PerfLogger::new("init static mi map");
        let loaded_vocab = init_vocab()?;
        log::info!("loaded clitics len {}", loaded_vocab.len());

        let mut vocab: HashMap<String, Arc<Vec<WorkMI>>> = HashMap::new();
        for (key, value) in loaded_vocab {
            let v = to_word_mi(value.as_str());
            vocab.insert(key, v);
        }
        vocab.shrink_to_fit();
        let res = StaticWords {
            vocab,
            numeric_mi: to_word_mi("M----d-"),
            x_mi: to_word_mi("X-"),
            email_mi: to_word_mi(MI_EMAIL),
            url_mi: to_word_mi(MI_URL),
        };
        Ok(res)
    }

    fn try_find(&self, w: &str) -> Option<Arc<Vec<WorkMI>>> {
        if is_number(w) {
            return Some(self.numeric_mi.clone());
        }
        if let Some(s) = self.vocab.get(w) {
            return Some(s.clone());
        }

        let mut finder = LinkFinder::new();
        finder.url_must_have_scheme(false);
        let links: Vec<_> = finder.links(w).collect();
        if links.len() == 1 {
            let link = &links[0];
            if link.start() == 0 && link.end() == w.len() {
                if matches!(link.kind(), linkify::LinkKind::Url) {
                    return Some(self.url_mi.clone());
                }
                if matches!(link.kind(), linkify::LinkKind::Email) {
                    return Some(self.email_mi.clone());
                }
            }
        }
        if starts_with_nonalpha_num(w) {
            return Some(self.x_mi.clone());
        }
        if w.len() > 1 && w.contains('%') {
            return Some(self.x_mi.clone());
        }
        None
    }
}

fn to_word_mi(s: &str) -> Arc<Vec<WorkMI>> {
    Arc::new(vec![WorkMI {
        lemma: None,
        mi: Some(s.to_string()),
    }])
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
                if word_info.is_word && word_info.mis.is_none() && word_info.mi.is_none() {
                    word_info.mis = self.try_find(&word_info.w.to_lowercase());
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
    for o in ["|", "\\", "*", "%", "^", "$", "•", "+", "§"] {
        res_map.insert(o.to_string(), "Tx".to_string());
    }
    for o in SYMBOLS.chars() {
        // do not overwrite above values
        res_map
            .entry(o.to_string())
            .or_insert_with(|| "Tx".to_string());
    }
    for o in MATH_SYMBOLS.chars() {
        // do not overwrite above values
        res_map
            .entry(o.to_string())
            .or_insert_with(|| "Tx".to_string());
    }
    Ok(res_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    fn test_try_find(input: &str, expected: Vec<String>) {
        let p = StaticWords::new().unwrap();
        let mut res: Vec<String> = Vec::new();
        let got = p.try_find(input);
        if got.is_none() {      
            assert_eq!(expected.len(), 0, "try_find '{}'", input);
            return;
        }
        for v in got.unwrap().iter().cloned() {
            res.push(v.mi.unwrap());
        }
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
        number: "10", vec!["M----d-".to_string()],
        coma: ",", vec!["Tc".to_string()],
        dot: ".", vec!["Tp".to_string()],
        some: "=", vec!["Tx".to_string()],
        with_dot: ".olia", vec!["X-".to_string()],
        with_percent: "olia%", vec!["X-".to_string()],
        with_percent_middle: "oli%a", vec!["X-".to_string()],
        with_percent_start: "%olia", vec!["X-".to_string()],
        url: "lrt.lt", vec!["Dl".to_string()],
        email: "a@aa.lt", vec!["De".to_string()],
        not_start: " lrt.lt", vec!["X-".to_string()],
        not_end: "lrt.lt ", vec![],
        several_links: "lrt.lt lrt.lt", vec![],
        word: "mama", vec![],
        symbol: "§", vec!["Tx".to_string()],
        symbol_math: "⅞", vec!["Tx".to_string()],
    );
}
