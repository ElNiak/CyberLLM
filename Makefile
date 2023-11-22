build:
	docker build --rm -t cyberllm .

run:
	docker run -it --rm -p 8888:8888 cyberllm