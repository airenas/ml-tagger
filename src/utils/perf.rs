use std::time::Instant;

pub struct PerfLogger {
    start: std::time::Instant,
    name: String,
}

impl Drop for PerfLogger {
    fn drop(&mut self) {
        tracing::info!(
            name = self.name,
            elapsed = format!("{:.2?}", self.start.elapsed()),
            "===== end"
        );
    }
}

impl PerfLogger {
    pub fn new(name: &str) -> PerfLogger {
        tracing::debug!(name, "===== start");
        let start = Instant::now();
        PerfLogger {
            start,
            name: name.to_string(),
        }
    }
}
