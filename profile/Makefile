#################################################################
version?=0.1.3-debug
l_url=localhost:8000/debug/pprof/heap
url=https://sinteze.intelektika.lt/tagger-ml/debug/pprof/heap
#url=http://tagger.worker1.mywire.org:8080/debug/pprof/heap
file=f1
base=f1
#################################################################
prepare/bin:
	docker run -it --rm -d --name ml-dev --entrypoint sh airenas/ml-tagger:$(version)
	docker cp ml-dev:/app/ml-tagger-ws ./ml-tagger-ws
	docker stop ml-dev
#################################################################
prepare/pproof:
	go install github.com/google/pprof@latest
#################################################################
get:
	curl $(url) > $(file).pb.gz
show:
	pprof -http=:8080 $(file).pb.gz
diff:
	pprof -http=:8080 -diff_base=$(base).pb.gz $(file).pb.gz
#################################################################
get/local:
	curl $(l_url) > $(file).pb.gz
show/local:
	cd .. && pprof -http=:8080 profile/$(file).pb.gz
#################################################################
