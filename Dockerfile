FROM tensorflow/tensorflow

ENV DEBIAN_FRONTEND="noninteractive"
ENV SNAPCRAFT_SETUP_CORE=1


RUN  apt-get update
RUN  apt update; apt-get --fix-missing -y install \ 
                # python3.9 \
                # python3.9-dev \
                # python3.9-venv \
                # python3-pip \
                xterm \
                git \
                curl \
                nano \
                libssl-dev

# RUN curl https://sh.rustup.rs -sSf | sh; source "$HOME/.cargo/env"
WORKDIR /app/cyberllm

ADD cyberllm /app/cyberllm

RUN cd cyberllm; pip install --upgrade pip; pip3 install wheel; pip3 install setuptools_rust; pip3 install .;