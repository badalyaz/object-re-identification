FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN mkdir /opt/project

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install --no-install-recommends -yy \
    python3 python3-pip python3-setuptools python3-wheel python3-dev wget git gcc

RUN pip3 install --upgrade pip setuptools
RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR /opt/project