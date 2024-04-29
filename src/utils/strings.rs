use std::collections::HashSet;

use once_cell::sync::Lazy;
use regex::Regex;
use url::Url;

pub fn is_number(w: &str) -> bool {
    w.replace(',', ".").parse::<f64>().is_ok()
}

pub fn is_url(w: &str) -> bool {
    static POP_EXT: Lazy<HashSet<&str>> = Lazy::new(|| {
        [
            "com", "org", "net", "edu", "gov", "lt", "lv", "ee", "fi", "ru", "pl", "de", "uk",
            "localhost", "dr", "es", "se"
        ]
        .iter()
        .cloned()
        .collect()
    });
    if is_number(w) {
        return false;
    }
    if w.contains(' ') {
        return false;
    }
    if w.contains('.') || w.contains("://") {
        if w.contains("://") {
            if Url::parse(w).is_ok() {
                return true;
            }
        } else {
            let res = Url::parse(("http://".to_string() + w).as_str());
            return match res {
                Ok(url) => {
                    if let Some(host) = url.host_str() {
                        if let Some(ext) = host.rsplit('.').next() {
                            if POP_EXT.contains(ext.to_lowercase().as_str()) {
                                return true;
                            }
                        }
                    }
                    return false;
                }
                Err(_) => false,
            };
        }
    }
    false
}

pub fn is_email(w: &str) -> bool {
    static RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$").unwrap()
    });
    RE.is_match(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_is_number() {
        assert_eq!(is_number("10"), true);
        assert_eq!(is_number("olia"), false);
        assert_eq!(is_number("."), false);
        assert_eq!(is_number(".0001"), true, "{}", 0.0001);
        assert_eq!(is_number("10,121212"), true, "{}", "10,121212");
        assert_eq!(is_number("10.121212"), true);
        assert_eq!(is_number("-10.121212"), true, "{}", -10.121212);
    }

    fn _test_is_url(w: &str, expected: bool) {
        assert_eq!(is_url(w), expected, "parse '{}'", w);
    }
    #[test]
    fn test_is_url() {
        _test_is_url("ftp://a:a@localhost:8080/aaa", true);
        _test_is_url("www.olia.lt/opa?aaa=10&bbb=132123 xxx", false);
        _test_is_url("olia", false);
        _test_is_url(".olia", false);
        _test_is_url("olia.", false);
        _test_is_url("10.121212", false);
        _test_is_url("olia.com", true);
        _test_is_url("www.olia.com", true);
        _test_is_url("www.olia.lt", true);
        _test_is_url("www.olia.lt/opa?aaa=10&bbb=132123", true);
        _test_is_url(
            "http://klcdocker.vdu.lt/morfdemo/api.lema/analyze/jis?human=true&origin=true",
            true,
        );
        _test_is_url("http://olia.com", true);
        _test_is_url("https://olia.com", true);
        _test_is_url("https://www.olia.co.uk", true);
        _test_is_url("tris.trys-LMT-K-septyni", false);
        _test_is_url("du.du-LMT-K", false);
    }

    #[test]
    fn test_is_email() {
        assert_eq!(is_email("oloa.o@olia.com"), true);
        assert_eq!(is_email("tata@tata.lt"), true);
        assert_eq!(is_email("."), false);
        assert_eq!(is_email("olia.com"), false);
        assert_eq!(is_email("@oooo.com"), false);
        assert_eq!(is_email("@oooo.com"), false);
    }
}
