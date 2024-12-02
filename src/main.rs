mod config;

use axum::extract::DefaultBodyLimit;
use chrono::NaiveDateTime;
use clap::Parser;
use config::make_file_path;
use ml_tagger::handlers::clean_cache::CacheData;
use ml_tagger::handlers::data::{self, Service, WorkMI};
use ml_tagger::handlers::{self};
use ml_tagger::processors::metrics::Metrics;
use ml_tagger::utils::perf::PerfLogger;
use ml_tagger::{processors, FN_CLITICS, FN_TAGS, FN_TAGS_FREQ};
use moka::future::Cache;
use moka::policy::EvictionPolicy;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tokio::time;
use tokio_util::sync::CancellationToken;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use tokio::signal::unix::{signal, SignalKind};
use ulid::Ulid;

use axum::{
    routing::{get, post},
    Router,
};

/// ML POS tagger http service
#[derive(Parser, Debug)]
#[command(version = env!("CARGO_APP_VERSION"), name = "ml-tagger-ws", about="Service for serving ML based POS tagger", 
    long_about = None, author="Airenas V.<airenass@gmail.com>")]
struct Args {
    /// Server port
    #[arg(long, env, default_value = "8000")]
    port: u16,
    /// Embeddings file
    #[arg(long, env = "EMBEDDINGS_FILE", required = true)]
    embeddings: String,
    /// ONNX file
    #[arg(long, env = "ONNX_FILE", required = true)]
    onnx: String,
    /// Data directory
    #[arg(long, env = "DATA_DIR", required = true)]
    data_dir: String,
    /// Lemma URL
    #[arg(long, env = "LEMMA_URL", required = true)]
    lemma_url: String,
    // Lex URL
    #[arg(long, env = "LEX_URL", required = true)]
    lex_url: String,
    /// Threads to use for ONNX inference
    #[arg(long, env = "ONNX_THREADS", required = false, default_value = "1")]
    onnx_threads: u16,
    /// Max items for embeddings cache
    #[arg(
        long,
        env = "EMBEDDINGS_CACHE",
        required = false,
        default_value = "50000"
    )]
    embeddings_cache: u64,
    /// Max items for lemma cache
    #[arg(long, env = "LEMMA_CACHE", required = false, default_value = "100000")]
    lemma_cache: u64,
    /// Key for cache cleaning
    #[arg(long, env = "CACHE_KEY", required = false, default_value = "")]
    cache_key: String,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::Layer::default().compact())
        .init();
    let args = Args::parse();
    if let Err(e) = main_int(args).await {
        log::error!("{}", e);
        return Err(e);
    }
    Ok(())
}

async fn main_int(cfg: Args) -> anyhow::Result<()> {
    let _perf_log = PerfLogger::new("loading service");
    tracing::info!("Starting ML Tagger service");
    tracing::info!(version = env!("CARGO_APP_VERSION"));
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
        }
        tracing::debug!("sending exit event");
        ct.cancel();
        tracing::debug!("expected drop tx_close");
    });

    let lexer = processors::lex::Lexer::new(&cfg.lex_url)?;
    let boxed_lexer: Box<dyn data::Processor + Send + Sync> = Box::new(lexer);

    let embeddings_cache: Cache<String, Arc<Vec<f32>>> = Cache::builder()
        .max_capacity(cfg.embeddings_cache)
        .eviction_policy(EvictionPolicy::tiny_lfu())
        .time_to_idle(Duration::from_secs(60 * 60 * 5)) // 5h
        .build();
    let lemma_cache: Cache<String, Arc<Vec<WorkMI>>> = Cache::builder()
        .max_capacity(cfg.lemma_cache)
        .eviction_policy(EvictionPolicy::tiny_lfu())
        .time_to_idle(Duration::from_secs(60 * 60 * 5)) // 5h
        .build();

    let (boxed_embedder, dims) = if cfg.embeddings.ends_with(".fifu") {
        let embedder = processors::embedding_ff::FinalFusionWrapper::new(
            &cfg.embeddings,
            embeddings_cache.clone(),
        )?;
        let dims = embedder.dims();
        let be: Box<dyn data::Processor + Send + Sync> = Box::new(embedder);
        (be, dims)
    } else {
        let embedder =
            processors::embedding::FastTextWrapper::new(&cfg.embeddings, embeddings_cache.clone())?;
        let dims = embedder.dims();
        let be: Box<dyn data::Processor + Send + Sync> = Box::new(embedder);
        (be, dims)
    };

    let onnx = processors::onnx::OnnxWrapper::new(&cfg.onnx, cfg.onnx_threads, dims)?;
    // let onnx = processors::ts::TSWrapper::new()?;
    let boxed_onnx: Box<dyn data::Processor + Send + Sync> = Box::new(onnx);

    let tags = processors::tags::TagsMapper::new(&make_file_path(&cfg.data_dir, FN_TAGS)?)?;
    let boxed_tags: Box<dyn data::Processor + Send + Sync> = Box::new(tags);

    let lw_mapper = processors::lemmatize_words::LemmatizeWordsMapper::new(
        &cfg.lemma_url,
        lemma_cache.clone(),
    )?;
    let boxed_lw_mapper: Box<dyn data::Processor + Send + Sync> = Box::new(lw_mapper);

    let clitics = processors::clitics::Clitics::new(&make_file_path(&cfg.data_dir, FN_CLITICS)?)?;
    let boxed_clitics: Box<dyn data::Processor + Send + Sync> = Box::new(clitics);

    let statics = processors::static_words::StaticWords::new()?;
    let boxed_statics: Box<dyn data::Processor + Send + Sync> = Box::new(statics);

    let restorer =
        processors::restorer::Restorer::new(&make_file_path(&cfg.data_dir, FN_TAGS_FREQ)?)?;
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

    let main_router = Router::new()
        .route("/live", get(handlers::live::handler))
        .route("/tag", post(handlers::tag::handler))
        .route("/tag-parsed", post(handlers::tag_parsed::handler))
        .with_state(srv.clone());

    let cache_key = if cfg.cache_key.is_empty() {
        let ulid = Ulid::new();
        ulid.to_string()
    } else {
        cfg.cache_key.clone()
    };
    let caches = CacheData {
        lemma_cache: lemma_cache.clone(),
        embeddings_cache: embeddings_cache.clone(),
    };
    let cache_path = format!("/clean-cache/{}", cache_key);
    tracing::info!(path = cache_path, "clean cache");
    let cache_router = Router::new()
        .route(&cache_path, post(handlers::clean_cache::handler))
        .with_state(Arc::new(caches));

    let app = Router::new()
        .merge(main_router)
        .merge(cache_router)
        .route("/metrics", get(handlers::metrics::handler))
        .layer((
            DefaultBodyLimit::max(1024 * 1024),
            TraceLayer::new_for_http(),
            TimeoutLayer::new(Duration::from_secs(50)),
        ));

    let metrics = Metrics::new(vec!["/tag-parsed".to_string(), "/tag".to_string()])?;
    let cp_metrics = metrics.clone();

    // let final_routes = routes
    //     .with(warp::log::custom(move |log| cp_metrics.observe(log)))
    //     .recover(errors::handle_rejection);

    let ct = cancel_token.clone();
    let embeddigs_cache_clone = embeddings_cache.clone();
    let lemma_cache_clone = lemma_cache.clone();
    let timer = tokio::task::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(5));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    tracing::trace!("check timer");
                    metrics.observe_cache("embeddings", embeddigs_cache_clone.entry_count());
                    metrics.observe_cache("lemma", lemma_cache_clone.entry_count());
                },
                _ = ct.cancelled() => {
                    break
                }
            }
        }
        tracing::info!("finished cache check timer");
    });

    let ct = cancel_token.clone();
    let cache_timer = tokio::task::spawn(async move {
        loop {
            let after = get_next_clear_run();
            tracing::info!(after = format!("{:?}", after), "next clear cache");
            tokio::select! {
                _ = time::sleep(after) => {
                    log::info!("clear cache");
                    embeddings_cache.invalidate_all();
                    lemma_cache.invalidate_all();
                },
                _ = ct.cancelled() => {
                    break
                }
            }
        }
        tracing::info!("finished cache clear timer");
    });

    std::mem::drop(_perf_log);
    tracing::info!(port = cfg.port, "serving ...");

    let listener = TcpListener::bind(format!("0.0.0.0:{}", cfg.port)).await?;

    let handle = axum_server::Handle::new();
    let shutdown_future = shutdown_signal_handle(handle.clone(), cancel_token.clone());
    tokio::spawn(shutdown_future);

    // Run the server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            cancel_token.cancelled().await;
        })
        .await?;

    // tokio::task::spawn(server).await?;
    timer.await?;
    cache_timer.await?;

    tracing::info!("Bye");
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

async fn shutdown_signal_handle(handle: axum_server::Handle, cancel_token: CancellationToken) {
    cancel_token.cancelled().await;
    tracing::trace!("Received termination signal shutting down");
    handle.graceful_shutdown(Some(Duration::from_secs(10)));
}
