use prometheus::HistogramOpts;
use prometheus::HistogramVec;
use prometheus::IntGaugeVec;
use prometheus::Opts;

#[derive(Debug, Clone)]
pub struct Metrics {
    http_perf: HistogramVec,
    start_path: Vec<String>,

    cache_sizes: IntGaugeVec,
}

impl Metrics {
    pub fn new(start_path: Vec<String>) -> anyhow::Result<Self> {
        let http_perf = HistogramVec::new(
            HistogramOpts::new(
                "http_response_time_seconds",
                "HTTP method response time in seconds.",
            ),
            &["path", "status"],
        )?;
        let cache_sizes = IntGaugeVec::new(
            Opts::new("cache_items", "Items count in the cache."),
            &["cache"],
        )?;

        prometheus::default_registry().register(Box::new(http_perf.clone()))?;
        prometheus::default_registry().register(Box::new(cache_sizes.clone()))?;

        Ok(Self {
            http_perf,
            start_path,
            cache_sizes,
        })
    }

    pub fn observe(&self, info: warp::log::Info) {
        for sp in self.start_path.iter() {
            if info.path().contains(sp) {
                self.http_perf
                    .with_label_values(&[&sp, info.status().as_str()])
                    .observe(info.elapsed().as_secs_f64());
                return;
            }
        }
    }

    pub fn observe_cache(&self, name: &str, count: u64) {
        self.cache_sizes
            .with_label_values(&[name])
            .set(count as i64);
    }
}
