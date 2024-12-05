####################################################################################
service=airenas/ml-tagger
debug_ext=
debug_file_ext=
####################################################################################
version?=dev
########### DOCKER ##################################################################
tag=$(service):$(version)

dbuild:
ifeq ($(debug),true)
	echo "DEBUG MODE"
	cd ../.. && docker buildx build -t $(tag)-debug --build-arg BUILD_VERSION=$(version) -f build/ml-tagger/Dockerfile.debug .
else
	cd ../.. && docker buildx build -t $(tag) --build-arg BUILD_VERSION=$(version) -f build/ml-tagger/Dockerfile .
endif	

dpush: dbuild
ifeq ($(debug),true)
	echo "DEBUG MODE"
	docker push $(tag)-debug
else
	docker push $(tag)
endif	

dscan: dbuild
	docker scan --accept-license $(tag)	
#####################################################################################
.PHONY: dbuild dpush
