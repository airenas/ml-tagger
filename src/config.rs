use clap::ArgMatches;

pub struct Config {
    pub port: u16,
    pub version: String,
    pub embeddings: String,
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
        Ok(Config {
            port,
            version: "dev".to_string(),
            embeddings: embeddings_file.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {}
