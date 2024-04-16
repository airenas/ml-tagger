pub fn is_number(w: &str) -> bool {
    w.replace(',', ".").parse::<f64>().is_ok()
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
}
