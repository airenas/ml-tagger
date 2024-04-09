use clap::ArgMatches;

pub struct Config {
    pub port: u16,
    pub version: String,
    pub embeddings: String,
    pub onnx: String,
    pub tags: String,
    pub lemma_url: String,
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
        let tags_file = match args.get_one::<String>("tags") {
            Some(v) => Ok(v),
            None => Err("no tags file"),
        }?;
        Ok(Config {
            port,
            version: "dev".to_string(),
            embeddings: embeddings_file.to_string(),
            onnx: onnx_file.to_string(),
            tags: tags_file.to_string(),
            lemma_url: "http://klcdocker.vdu.lt/morfdemo/api.lema/analyze".to_string()
        })
    }
}

#[cfg(test)]
mod tests {}
