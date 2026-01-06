use anyhow::Ok;
use async_trait::async_trait;
use linkify::LinkFinder;

use crate::handlers::data::{Processor, WorkContext};
use crate::utils::perf::PerfLogger;

pub struct Finder {}

impl Finder {
    pub fn new() -> anyhow::Result<Finder> {
        tracing::info!("init url finder");
        let res = Finder {};
        Ok(res)
    }
}

#[async_trait]
impl Processor for Finder {
    async fn process(&self, ctx: &mut WorkContext) -> anyhow::Result<()> {
        let _perf_log = PerfLogger::new("url finder");
        let txt = &ctx.text;
        let mut finder = LinkFinder::new();
        finder.url_must_have_scheme(false);

        let links: Vec<_> = finder.links(txt).collect();
        for link in links {
            let start = link.start();
            let end = link.end();
            ctx.links.push(crate::handlers::data::Link {
                start,
                end,
                kind: match link.kind() {
                    linkify::LinkKind::Url => linkify::LinkKind::Url,
                    linkify::LinkKind::Email => linkify::LinkKind::Email ,
                    _ => linkify::LinkKind::Url,
                },
            });

            tracing::trace!("found url from {} to {}: {}", start, end, &txt[start..end]);
        }

        Ok(())
    }
}
