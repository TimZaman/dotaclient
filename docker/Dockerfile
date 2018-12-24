FROM ubuntu:18.04
MAINTAINER Tim Zaman <timbobel@gmail.com>

RUN apt-get -q update \
 && apt-get install -y \
    python3.7 \
    python3.7-distutils \
    curl \
 && \
    apt-get -y upgrade && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/{apt,dpkg,cache,log}/ \
 && curl -s https://bootstrap.pypa.io/get-pip.py | python3.7

RUN pip3.7 install --user dotaservice==0.1.5 torch tensorboardX google-cloud-storage

RUN mkdir /root/dotaclient

WORKDIR /root/dotaclient

COPY __init__.py client.py dotaoptimizer.py policy.py /root/dotaclient/
COPY protos/ /root/dotaclient/protos/

