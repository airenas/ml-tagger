use axum::extract::DefaultBodyLimit;
use clap::Parser;
use ml_tagger::config::make_file_path;
use ml_tagger::handlers::pprof;
use ml_tagger::utils::perf::PerfLogger;
use ml_tagger::{processors, FN_CLITICS, FN_TAGS, FN_TAGS_FREQ};
use std::time::Duration;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use tokio::signal::unix::{signal, SignalKind};

use axum::{routing::get, Router};

/// ML POS tagger http service
#[derive(Parser, Debug)]
#[command(version = env!("CARGO_APP_VERSION"), name = "inspect-dict", about="Test dictionary", 
    long_about = None, author="Airenas V.<airenass@gmail.com>")]
struct Args {
    /// Server port
    #[arg(long, env, default_value = "8000")]
    port: u16,
    /// Data directory
    #[arg(long, env = "DATA_DIR", required = true)]
    data_dir: String,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    // console_subscriber::init();
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
    tracing::info!(dir = cfg.data_dir);

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

    let _tags = processors::tags::TagsMapper::new(&make_file_path(&cfg.data_dir, FN_TAGS)?)?;

    let _clitics = processors::clitics::Clitics::new(&make_file_path(&cfg.data_dir, FN_CLITICS)?)?;

    let _statics = processors::static_words::StaticWords::new()?;

    let _restorer =
        processors::restorer::Restorer::new(&make_file_path(&cfg.data_dir, FN_TAGS_FREQ)?)?;

    let helper_router = axum::Router::new().route("/debug/pprof/heap", get(pprof::handler));

    let app = Router::new().merge(helper_router).layer((
        DefaultBodyLimit::max(1024 * 1024),
        TraceLayer::new_for_http(),
        TimeoutLayer::new(Duration::from_secs(50)),
    ));

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

    tracing::info!("Bye");
    Ok(())
}

async fn shutdown_signal_handle(handle: axum_server::Handle, cancel_token: CancellationToken) {
    cancel_token.cancelled().await;
    tracing::trace!("Received termination signal shutting down");
    handle.graceful_shutdown(Some(Duration::from_secs(10)));
}
