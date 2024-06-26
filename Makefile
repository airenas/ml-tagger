-include Makefile.options
log?=INFO
###############################################################################
run:
	RUST_LOG=$(log) cargo run --bin ml-tagger-ws -- --embeddings ${EMBEDDINGS_FILE} --onnx ${ONNX_FILE} --data_dir ${DATA_DIR} --lemma_url "${LEMMA_URL}" --lex_url "${LEX_URL}"
.PHONY: run
###############################################################################
run/build: build/local
	RUST_LOG=$(log) target/release/ml-tagger-ws --embeddings ${EMBEDDINGS_FILE} --onnx ${ONNX_FILE} --data_dir ${DATA_DIR} --lemma_url "${LEMMA_URL}" --lex_url "${LEX_URL}"
.PHONY: run/build
###############################################################################
build/local: 
	cargo build --release
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
