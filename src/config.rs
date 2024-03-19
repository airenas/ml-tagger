use clap::ArgMatches;

pub struct Config {
    pub port: u16,
    pub version: String,
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
        Ok(Config {
            port,
            version: "dev".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {}
