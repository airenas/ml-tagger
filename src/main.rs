mod config;

use clap::Arg;
use ml_tagger::handlers::data::Service;
use ml_tagger::handlers::{self, errors};
use std::process;
use std::{error::Error, sync::Arc};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use warp::Filter;

use clap::Command;
use config::Config;
use tokio::signal::unix::{signal, SignalKind};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let cfg = app_config().unwrap_or_else(|err| {
        log::error!("problem parsing arguments: {err}");
        process::exit(1)
    });

    log::info!("Starting ML Tagger service");
    log::info!("Version      : {}", cfg.version);
    log::info!("Port         : {}", cfg.port);

    let cancel_token = CancellationToken::new();

    let ct = cancel_token.clone();

    tokio::spawn(async move {
        let mut int_stream = signal(SignalKind::interrupt()).unwrap();
        let mut term_stream = signal(SignalKind::terminate()).unwrap();
        tokio::select! {
            _ = int_stream.recv() => log::info!("Exit event int"),
            _ = term_stream.recv() => log::info!("Exit event term"),
            // _ = rx_exit_indicator.recv() => log::info!("Exit event from some loader"),
        }
        log::debug!("sending exit event");
        ct.cancel();
        log::debug!("expected drop tx_close");
    });

    // let pool = deadpool_redis::Config::from_url(&cfg.redis_url)
    //     .create_pool(Some(Runtime::Tokio1))
    //     .unwrap_or_else(|err| {
    //         log::error!("redis poll init: {err}");
    //         process::exit(1)
    //     });

    // let db = RedisClient::new(pool).await.unwrap_or_else(|err| {
    //     log::error!("redis client init: {err}");
    //     process::exit(1)
    // });

    let srv = Arc::new(RwLock::new(Service {}));

    let live_route = warp::get()
        .and(warp::path("live"))
        .and(with_service(srv.clone()))
        .and_then(handlers::live::handler);
    let tag_route = warp::post()
        .and(warp::path("tag"))
        .and(json_body())
        .and(with_service(srv.clone()))
        .and_then(handlers::tag::handler);

    let routes = live_route
        .or(tag_route)
        .with(warp::cors().allow_any_origin())
        .recover(errors::handle_rejection);

    let ct = cancel_token.clone();
    let (_, server) =
        warp::serve(routes).bind_with_graceful_shutdown(([0, 0, 0, 0], cfg.port), async move {
            ct.cancelled().await;
        });

    log::info!("wait for server to finish");
    tokio::task::spawn(server).await.unwrap_or_else(|err| {
        log::error!("{err}");
        process::exit(1);
    });

    log::info!("Bye");
    Ok(())
}

fn json_body() -> impl Filter<Extract = (Vec<Vec<String>>,), Error = warp::Rejection> + Clone {
    warp::body::content_length_limit(1024 * 1024).and(warp::body::json())
}

fn with_service(
    srv: Arc<RwLock<Service>>,
) -> impl Filter<Extract = (Arc<RwLock<Service>>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || srv.clone())
}

fn app_config() -> Result<Config, String> {
    let app_version = option_env!("CARGO_APP_VERSION").unwrap_or("dev");

    let cmd = Command::new("ml-tagger-ws")
        .version(app_version)
        .author("Airenas V.<airenass@gmail.com>")
        .about("Service for serving ML based POS tagger")
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("Service port")
                .env("PORT")
                .default_value("8000"),
        )
        .get_matches();
    let mut config = Config::build(&cmd)?;
    config.version = app_version.into();
    Ok(config)
}
