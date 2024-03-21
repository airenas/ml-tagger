use std::time::Instant;

pub struct PerfLogger {
    start: std::time::Instant,
    name: String,
}

impl Drop for PerfLogger {
    fn drop(&mut self) {
        log::info!("===== end: {} in {:.2?}", self.name, self.start.elapsed());
    }
}

impl PerfLogger {
    pub fn new(name: &str) -> PerfLogger {
        log::debug!("start: {name}");
        let start = Instant::now();
        PerfLogger {
            start,
            name: name.to_string(),
        }
    }
}
