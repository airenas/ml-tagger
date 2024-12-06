use super::data::ApiResult;
use super::error::ApiError;

pub async fn handler() -> ApiResult<Vec<u8>> {
    tracing::debug!("pprof handler");
    let mut prof_ctl = jemalloc_pprof::PROF_CTL
        .as_ref()
        .ok_or_else( ApiError::PProfNotActivated)?
        .lock()
        .await;
    require_profiling_activated(&prof_ctl)?;
    let pprof = prof_ctl.dump_pprof().map_err(|err| anyhow::anyhow!(err))?;
    Ok(pprof)
}

/// Checks whether jemalloc profiling is activated an returns an error response if not.
fn require_profiling_activated(prof_ctl: &jemalloc_pprof::JemallocProfCtl) -> Result<(), ApiError> {
    if !prof_ctl.activated() {
        return Err(ApiError::PProfNotActivated());
    }
    Ok(())
}
