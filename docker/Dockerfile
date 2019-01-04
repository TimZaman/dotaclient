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

RUN pip3.7 install --user torch==1.0.0

RUN pip3.7 install --user tensorboardX==1.5.0 google-cloud-storage==1.13.0 pika==0.12.0 aioamqp==0.12.0 grpcio==1.16.1

RUN pip3.7 install --user dotaservice==0.3.0

RUN mkdir /root/dotaclient

WORKDIR /root/dotaclient

COPY __init__.py agent.py optimizer.py policy.py distributed.py /root/dotaclient/
