use once_cell::sync::Lazy;
use regex::Regex;

pub fn is_number(w: &str) -> bool {
    w.replace(',', ".").parse::<f64>().is_ok()
}

pub fn is_url(w: &str) -> bool {
    if is_number(w) {
        return false;
    }
    static RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^(((http|ftp)s?://)?([\w-]+(\.[\w-]+)+\.?|localhost)(:\d+)?(/\S*)?)$").unwrap());
    RE.is_match(w)
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
        assert_eq!(is_url(w), expected, "{}", w);
    }
    #[test]
    fn test_is_url() {
        _test_is_url("olia", false);
        _test_is_url(".olia", false);
        _test_is_url("olia.", false);
        _test_is_url("10.121212", false);
        _test_is_url("www.olia.lt/opa?aaa=10&bbb=132123 xxx", false);
        _test_is_url("olia.com", true);
        _test_is_url("www.olia.com", true);
        _test_is_url("www.olia.lt", true);
        _test_is_url("www.olia.lt/opa?aaa=10&bbb=132123", true);
        _test_is_url("http://klcdocker.vdu.lt/morfdemo/api.lema/analyze/jis?human=true&origin=true", true);
        _test_is_url("http://olia.com", true);
        _test_is_url("https://olia.com", true);
        _test_is_url("https://www.olia.co.uk", true);
        _test_is_url("ftp://localhost:8080/aaa", true);
    }

    #[test]
    fn test_is_email() {
        assert_eq!(is_email("oloa.o@olia.com"), true);
        assert_eq!(is_email("tata@tata.lt"), true);
        assert_eq!(is_email("."), false);
        assert_eq!(is_email("olia.com"), false);
        assert_eq!(is_email("@oooo.com"), false);
    }
}
