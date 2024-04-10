use std::path::Path;

use clap::ArgMatches;

pub struct Config {
    pub port: u16,
    pub version: String,
    pub embeddings: String,
    pub onnx: String,
    pub tags: String,
    pub lemma_url: String,
    pub data_dir: String,
    pub clitics: String,
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
            None => Err("no data_dir"),
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
        Ok(Config {
            port,
            version: "dev".to_string(),
            embeddings: embeddings_file.to_string(),
            onnx: onnx_file.to_string(),
            tags,
            data_dir: data_dir.to_string(),
            lemma_url: "http://klcdocker.vdu.lt/morfdemo/api.lema/analyze".to_string(),
            clitics,
        })
    }
}

#[cfg(test)]
mod tests {}
