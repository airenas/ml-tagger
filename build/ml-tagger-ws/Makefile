####################################################################################
service=airenas/ml-tagger-ws
version?=dev
########### DOCKER ##################################################################
tag=$(service):$(version)

dbuild:
	cd ../.. && docker buildx build -t $(tag) --progress=plain --build-arg BUILD_VERSION=$(version) -f build/ml-tagger-ws/Dockerfile .

dpush: dbuild
	docker push $(tag)

dscan: dbuild
	docker scan --accept-license $(tag)	
#####################################################################################
.PHONY: dbuild dpush
