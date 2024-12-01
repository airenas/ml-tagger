use std::path::Path;

pub fn make_file_path(dir: &str, file: &str) -> anyhow::Result<String> {
        Path::new(dir)
            .join(file)
            .into_os_string()
            .into_string()
            .map_err(|e| anyhow::anyhow!("can't prepare file {file}: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_file_path() {
        let dir = "/tmp";
        let file = "test.txt";
        let res = make_file_path(dir, file);
        assert!(res.is_ok());
        assert_eq!(res.unwrap(), "/tmp/test.txt");
    }
}

