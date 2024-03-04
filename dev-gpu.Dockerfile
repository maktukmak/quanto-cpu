FROM python:3.10
USER root
WORKDIR /
SHELL ["/bin/bash", "-c"]


RUN echo "export http_proxy=http://proxy-chain.intel.com:911/ && export https_proxy=http://proxy-chain.intel.com:912/" > /etc/profile && cat /etc/profile

RUN apt update
RUN pip config set global.extra-index-url "https://pypi.org/simple"
RUN pip install --upgrade pip

RUN apt-get update && apt install software-properties-common -y 
RUN apt install -y gcc-12 g++-12 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

RUN pip install mkl

#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install torch torchvision torchaudio

