-include Makefile.options
log?=INFO
###############################################################################
run:
	RUST_LOG=$(log) cargo run --bin ml-tagger-ws -- --embeddings ${EMBEDDINGS_FILE} --onnx ${ONNX_FILE} --data-dir ${DATA_DIR} --lemma-url "${LEMMA_URL}" --lex-url "${LEX_URL}" --onnx-threads 6
.PHONY: run
###############################################################################
run/build: build/local
	RUST_LOG=$(log) target/release/ml-tagger-ws --embeddings ${EMBEDDINGS_FILE} --onnx ${ONNX_FILE} --data-dir ${DATA_DIR} --lemma-url "${LEMMA_URL}" --lex-url "${LEX_URL}" --onnx-threads 6
.PHONY: run/build
run/build/debug: build/debug
	RUST_LOG=$(log) target/debug/ml-tagger-ws --embeddings ${EMBEDDINGS_FILE} --onnx ${ONNX_FILE} --data-dir ${DATA_DIR} --lemma-url "${LEMMA_URL}" --lex-url "${LEX_URL}" --onnx-threads 6
.PHONY: run/build
run/trace: 
	RUSTFLAGS="--cfg tokio_unstable" RUST_LOG=$(log),tokio=trace cargo run --features=profiling --bin ml-tagger-ws -- --embeddings ${EMBEDDINGS_FILE} --onnx ${ONNX_FILE} --data-dir ${DATA_DIR} --lemma-url "${LEMMA_URL}" --lex-url "${LEX_URL}" --onnx-threads 6 --cache-key=1
.PHONY: run/build
###############################################################################
run/trace/inspect: 
	RUST_LOG=$(log) cargo run --features=profiling --bin inspect-dict -- --data-dir ${DATA_DIR}
.PHONY: run/build
###############################################################################
build/local: 
	cargo build --release
.PHONY: build/local
build/debug: 
	cargo build --features profiling
.PHONY: build/local
###############################################################################
test/unit:
	RUST_LOG=DEBUG cargo test --no-fail-fast
.PHONY: test/unit		
test/lint:
	@cargo clippy -V
	cargo clippy --all-targets --all-features -- -D warnings
.PHONY: test/lint	
###############################################################################
clean:
	rm -r -f target
.PHONY: clean

# test/service:
	# time for i in (seq 1 500)
    # 	curl -X POST "localhost:8000/tag?debug=false" -H "Content-Type: application/json" -d '[["mama", "olia", "."], ["haha", "tada", "olia", ".", "haha", "tada", "olia", ".","haha", "tada", "olia", ".","haha", "tada", "olia", ".","haha", "tada", "olia", ".","haha", "tada", "olia", "."]]' | jq .
  	# end
