#########################################################################################
## docker will invoke this file from ../.. dir in order to access code
#########################################################################################
FROM rust:1.77-buster as builder

# RUN apk --update add openssl-dev musl-dev

ARG BUILD_VERSION=0.1

WORKDIR /src/

COPY ./ /src

RUN --mount=type=cache,target=/usr/local/cargo/registry \
      CARGO_APP_VERSION=$BUILD_VERSION cargo build --release

RUN mkdir -p /onnx && find /src/target/ -name "*libonnxruntime.so.*" -exec cp "{}" /onnx/ \;
#########################################################################################
FROM debian:buster-slim as runner

RUN apt-get update && apt-get -y install openssl ca-certificates

ENV UID=999

ARG BUILD_VERSION=0.1

WORKDIR /app
ENV LD_LIBRARY_PATH=/onnx
EXPOSE 8000

RUN addgroup --gid $UID app && useradd -r -M -u $UID -g app app   
RUN mkdir -p /app && chown -R app:app /app   

COPY --from=builder /src/target/release/ml-tagger-ws /app/
COPY --from=builder /onnx/* /onnx/

RUN chown app:app /app/* /app
 
USER app

ENTRYPOINT ["./ml-tagger-ws"]
