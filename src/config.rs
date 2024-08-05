use std::path::Path;

use clap::ArgMatches;

pub struct Config {
    pub port: u16,
    pub version: String,
    pub embeddings: String,
    pub onnx: String,
    pub tags: String,
    pub lemma_url: String,
    pub lex_url: String,
    pub clitics: String,
    pub frequencies: String,
    pub onnx_threads: i16,
    pub embeddings_cache: u64,
    pub lemma_cache: u64,
    pub cache_key: String,
}

impl Config {
    pub fn build(args: &ArgMatches) -> Result<Config, String> {
        let port = match args.get_one::<String>("port") {
            Some(v) => v
                .trim()
                .parse::<u16>()
                .map_err(|e| -> String { format!("can't parse port: `{v}`, {e}") }),
            None => Err(String::from("no port provided")),
        }?;
        let embeddings_file = match args.get_one::<String>("embeddings") {
            Some(v) => Ok(v),
            None => Err("no embeddings file"),
        }?;
        let onnx_file = match args.get_one::<String>("onnx") {
            Some(v) => Ok(v),
            None => Err("no onnx file"),
        }?;
        let data_dir = match args.get_one::<String>("data_dir") {
            Some(v) => Ok(v),
            None => Err("no data dir"),
        }?;
        let lemma_url = match args.get_one::<String>("lemma_url") {
            Some(v) => Ok(v),
            None => Err("no lemma url"),
        }?;
        let lex_url = match args.get_one::<String>("lex_url") {
            Some(v) => Ok(v),
            None => Err("no lex url"),
        }?;
        let onnx_threads = match args.get_one::<String>("onnx_threads") {
            Some(v) => v
                .trim()
                .parse::<i16>()
                .map_err(|e| -> String { format!("can't parse onnx_threads: `{v}`, {e}") }),
            None => Err(String::from("no onnx_threads provided")),
        }?;
        let cache_key = match args.get_one::<String>("cache_key") {
            Some(v) => Ok(v),
            None => Err("no cache_key"),
        }?;
        let clitics = Path::new(data_dir)
            .join("clitics")
            .into_os_string()
            .into_string()
            .map_err(|e| -> String { format!("can't prepare file: {e:?}") })?;
        let tags = Path::new(data_dir)
            .join("tags")
            .into_os_string()
            .into_string()
            .map_err(|e| -> String { format!("can't prepare file: {e:?}") })?;
        let frequencies = Path::new(data_dir)
            .join("tags_freq")
            .into_os_string()
            .into_string()
            .map_err(|e| -> String { format!("can't prepare file: {e:?}") })?;
        let embeddings_cache = match args.get_one::<String>("embeddings_cache") {
            Some(v) => v
                .trim()
                .parse::<u64>()
                .map_err(|e| -> String { format!("can't parse embeddings_cache: `{v}`, {e}") }),
            None => Err(String::from("no embeddings_cache provided")),
        }?;
        let lemma_cache: u64 = match args.get_one::<String>("lemma_cache") {
            Some(v) => v
                .trim()
                .parse::<u64>()
                .map_err(|e| -> String { format!("can't parse lemma_cache: `{v}`, {e}") }),
            None => Err(String::from("no lemma_cache provided")),
        }?;
        Ok(Config {
            port,
            version: "dev".to_string(),
            embeddings: embeddings_file.to_string(),
            onnx: onnx_file.to_string(),
            tags,
            lemma_url: lemma_url.to_string(),
            lex_url: lex_url.to_string(),
            clitics,
            frequencies,
            onnx_threads,
            embeddings_cache,
            lemma_cache,
            cache_key: cache_key.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {}
