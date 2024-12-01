mod config;

use chrono::NaiveDateTime;
use clap::Arg;
use ml_tagger::handlers::data::{self, Service, TagParams, WorkMI};
use ml_tagger::handlers::{self, errors};
use ml_tagger::processors;
use ml_tagger::processors::metrics::Metrics;
use ml_tagger::utils::perf::PerfLogger;
use moka::future::Cache;
use moka::policy::EvictionPolicy;
use std::process;
use std::time::Duration;
use std::{error::Error, sync::Arc};
use tokio::sync::RwLock;
use tokio::time;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use warp::Filter;

use clap::Command;
use config::Config;
use tokio::signal::unix::{signal, SignalKind};
use ulid::Ulid;

// async fn main() -> anyhow::Result<()> {
//     tracing_subscriber::registry()
//         .with(tracing_subscriber::EnvFilter::from_default_env())
//         .with(tracing_subscriber::fmt::Layer::default().compact())
//         .init();
//     let args = Args::parse();
//     if let Err(e) = main_int(args).await {
//         log::error!("{}", e);
//         return Err(e);
//     }
//     Ok(())
// }

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::Layer::default().compact())
        .init();

    let _perf_log = PerfLogger::new("loading service");
    let cfg = app_config().unwrap_or_else(|err| {
        log::error!("problem parsing arguments: {err}");
        process::exit(1)
    });

    tracing::info!("Starting ML Tagger service");
    tracing::info!(version = cfg.version);
    tracing::info!(port = cfg.port);
    tracing::info!(cache = cfg.embeddings_cache);
    tracing::info!(lemma_cache = cfg.lemma_cache);

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

    let lexer = processors::lex::Lexer::new(&cfg.lex_url)?;
    let boxed_lexer: Box<dyn data::Processor + Send + Sync> = Box::new(lexer);

    let embeddigs_cache: Cache<String, Arc<Vec<f32>>> = Cache::builder()
        .max_capacity(cfg.embeddings_cache)
        .eviction_policy(EvictionPolicy::tiny_lfu())
        .time_to_idle(Duration::from_secs(60 * 60 * 5)) // 5h
        .build();
    let lemma_cache: Cache<String, Arc<Vec<WorkMI>>> = Cache::builder()
        .max_capacity(cfg.lemma_cache)
        .eviction_policy(EvictionPolicy::tiny_lfu())
        .time_to_idle(Duration::from_secs(60 * 60 * 5)) // 5h
        .build();

    let onnx = processors::onnx::OnnxWrapper::new(&cfg.onnx, cfg.onnx_threads)?;
    // let onnx = processors::ts::TSWrapper::new()?;
    let boxed_onnx: Box<dyn data::Processor + Send + Sync> = Box::new(onnx);

    let embedder =
        processors::embedding::FastTextWrapper::new(&cfg.embeddings, embeddigs_cache.clone())?;
    let boxed_embedder: Box<dyn data::Processor + Send + Sync> = Box::new(embedder);

    let tags = processors::tags::TagsMapper::new(&cfg.tags)?;
    let boxed_tags: Box<dyn data::Processor + Send + Sync> = Box::new(tags);

    let lw_mapper = processors::lemmatize_words::LemmatizeWordsMapper::new(
        &cfg.lemma_url,
        lemma_cache.clone(),
    )?;
    let boxed_lw_mapper: Box<dyn data::Processor + Send + Sync> = Box::new(lw_mapper);

    let clitics = processors::clitics::Clitics::new(&cfg.clitics)?;
    let boxed_clitics: Box<dyn data::Processor + Send + Sync> = Box::new(clitics);

    let statics = processors::static_words::StaticWords::new()?;
    let boxed_statics: Box<dyn data::Processor + Send + Sync> = Box::new(statics);

    let restorer = processors::restorer::Restorer::new(&cfg.frequencies)?;
    let boxed_restorer: Box<dyn data::Processor + Send + Sync> = Box::new(restorer);

    let srv = Arc::new(RwLock::new(Service {
        calls: 0,
        embedder: boxed_embedder,
        onnx: boxed_onnx,
        tag_mapper: boxed_tags,
        lemmatize_words_mapper: boxed_lw_mapper,
        clitics: boxed_clitics,
        restorer: boxed_restorer,
        static_words: boxed_statics,
        lexer: boxed_lexer,
    }));

    let live_route = warp::get()
        .and(warp::path("live"))
        .and(with_service(srv.clone()))
        .and_then(handlers::live::handler);
    let tag_route = warp::post()
        .and(warp::path("tag"))
        .and(warp::query::<TagParams>())
        .and(warp::body::content_length_limit(1024 * 1024))
        .and(warp::body::bytes())
        .and(with_service(srv.clone()))
        .and_then(handlers::tag::handler);
    let tag_parsed_route = warp::post()
        .and(warp::path("tag-parsed"))
        .and(warp::query::<TagParams>())
        .and(json_body())
        .and(with_service(srv.clone()))
        .and_then(handlers::tag_parsed::handler);
    let metrics_route = warp::get()
        .and(warp::path("metrics"))
        .and_then(handlers::metrics::handler);
    let l_cache_clone = lemma_cache.clone();
    let e_cache_clone = embeddigs_cache.clone();

    let cache_key = if cfg.cache_key.is_empty() {
        let ulid = Ulid::new();
        ulid.to_string()
    } else {
        cfg.cache_key.clone()
    };

    let cache_path = format!("clean-cache/{}", cache_key);
    log::info!("clean cache path: {}", cache_path);

    let clean_cache_route = warp::post()
        .and(warp::path("clean-cache"))
        .and(warp::path(cache_key))
        .and(warp::any().map(move || l_cache_clone.clone()))
        .and(warp::any().map(move || e_cache_clone.clone()))
        .and_then(handlers::clean_cache::handler);

    let metrics = Metrics::new(vec!["/tag-parsed".to_string(), "/tag".to_string()])?;
    let cp_metrics = metrics.clone();

    let routes = live_route
        .or(metrics_route)
        .or(tag_parsed_route)
        .or(tag_route)
        .or(clean_cache_route);

    let final_routes = routes
        .with(warp::cors().allow_any_origin())
        .with(warp::log::custom(move |log| cp_metrics.observe(log)))
        .recover(errors::handle_rejection);

    let ct = cancel_token.clone();
    let (_, server) = warp::serve(final_routes).bind_with_graceful_shutdown(
        ([0, 0, 0, 0], cfg.port),
        async move {
            ct.cancelled().await;
        },
    );

    let ct = cancel_token.clone();
    let embeddigs_cache_clone = embeddigs_cache.clone();
    let lemma_cache_clone = lemma_cache.clone();
    let timer = tokio::task::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(5));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    log::debug!("check timer");
                    metrics.observe_cache("embeddings", embeddigs_cache_clone.entry_count());
                    metrics.observe_cache("lemma", lemma_cache_clone.entry_count());
                },
                _ = ct.cancelled() => {
                    break
                }
            }
        }
        log::info!("finished cache check timer");
    });

    let ct = cancel_token.clone();
    let cache_timer = tokio::task::spawn(async move {
        loop {
            let after = get_next_clear_run();
            log::info!("next clear cache after: {:?}", after);
            tokio::select! {
                _ = time::sleep(after) => {
                    log::info!("clear cache");
                    embeddigs_cache.invalidate_all();
                    lemma_cache.invalidate_all();
                },
                _ = ct.cancelled() => {
                    break
                }
            }
        }
        log::info!("finished cache clear timer");
    });

    std::mem::drop(_perf_log);
    log::info!("Serving. waiting for server to finish");
    tokio::task::spawn(server).await.unwrap_or_else(|err| {
        log::error!("{err}");
        process::exit(1);
    });
    timer.await.unwrap_or_else(|err| {
        log::error!("{err}");
        process::exit(1);
    });
    cache_timer.await.unwrap_or_else(|err| {
        log::error!("{err}");
        process::exit(1);
    });

    log::info!("Bye");
    Ok(())
}

fn get_next_clear_run() -> time::Duration {
    let now = chrono::Local::now().naive_utc();
    let today = now.date();
    let mut next_day_3_am =
        NaiveDateTime::new(today, chrono::NaiveTime::from_hms_opt(3, 0, 0).unwrap());
    if now > next_day_3_am {
        next_day_3_am += chrono::Duration::days(1);
    }
    next_day_3_am.signed_duration_since(now).to_std().unwrap()
}

fn with_service(
    srv: Arc<RwLock<Service>>,
) -> impl Filter<Extract = (Arc<RwLock<Service>>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || srv.clone())
}

fn json_body() -> impl Filter<Extract = (Vec<Vec<String>>,), Error = warp::Rejection> + Clone {
    warp::body::content_length_limit(1024 * 1024).and(warp::body::json())
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
        .arg(
            Arg::new("embeddings")
                .long("embeddings")
                .value_name("EMBEDDINGS_FILE")
                .env("EMBEDDINGS_FILE")
                .help("Embeddings file")
                .required(true),
        )
        .arg(
            Arg::new("onnx")
                .long("onnx")
                .value_name("ONNX_FILE")
                .env("ONNX_FILE")
                .help("Onnx file")
                .required(true),
        )
        .arg(
            Arg::new("data_dir")
                .long("data_dir")
                .value_name("DATA_DIR")
                .env("DATA_DIR")
                .help("Data directory")
                .required(true),
        )
        .arg(
            Arg::new("lemma_url")
                .long("lemma_url")
                .value_name("LEMMA_URL")
                .env("LEMMA_URL")
                .help("Lemma URL")
                .required(true),
        )
        .arg(
            Arg::new("lex_url")
                .long("lex_url")
                .value_name("LEX_URL")
                .env("LEX_URL")
                .help("Lex URL")
                .required(true),
        )
        .arg(
            Arg::new("onnx_threads")
                .long("onnx_threads")
                .value_name("ONNX_THREADS")
                .env("ONNX_THREADS")
                .default_value("1")
                .help("Threads to use for ONNX inference")
                .required(false),
        )
        .arg(
            Arg::new("embeddings_cache")
                .long("embeddings_cache")
                .value_name("EMBEDDINGS_CACHE")
                .env("EMBEDDINGS_CACHE")
                .default_value("50000")
                .help("Max items for embeddings cache")
                .required(false),
        )
        .arg(
            Arg::new("lemma_cache")
                .long("lemma_cache")
                .value_name("LEMMA_CACHE")
                .env("LEMMA_CACHE")
                .default_value("100000")
                .help("Max items for lemma cache")
                .required(false),
        )
        .arg(
            Arg::new("cache_key")
                .long("cache_key")
                .value_name("CACHE_KEY")
                .env("CACHE_KEY")
                .default_value("")
                .help("Key for cache cleaning")
                .required(false),
        )
        .get_matches();
    let mut config = Config::build(&cmd)?;
    config.version = app_version.into();
    Ok(config)
}
