#########################################################################################
## docker will invoke this file from ../.. dir in order to access code
#########################################################################################
FROM rust:1.91-bookworm AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends  binutils

ARG BUILD_VERSION=0.1

ENV RUSTFLAGS="-C target-cpu=haswell -C opt-level=3 -C codegen-units=1"
ENV CARGO_PROFILE_RELEASE_DEBUG=false

WORKDIR /src/

COPY ./ /src

RUN --mount=type=cache,target=/usr/local/cargo/registry \
      CARGO_APP_VERSION=$BUILD_VERSION cargo build --release
RUN strip /src/target/release/ml-tagger-ws      
#########################################################################################
FROM debian:bookworm-slim AS runner

RUN apt-get update && \
    apt-get install -y --no-install-recommends  binutils \
    ca-certificates \
    openssl && \
    rm -rf /var/lib/apt/lists/*

ENV UID=999

ARG BUILD_VERSION=0.1

WORKDIR /app
EXPOSE 8000

RUN addgroup --gid $UID app && useradd -r -M -u $UID -g app app   
COPY LICENSE /licenses/LICENSE
RUN mkdir -p /app && chown -R app:app /app   

COPY --from=builder /src/target/release/ml-tagger-ws /app/

RUN chown app:app /app/* /app
 
USER app

ENTRYPOINT ["./ml-tagger-ws"]
