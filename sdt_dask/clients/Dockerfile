FROM python:3.10.11 as base

RUN apt-get update -qq
RUN apt-get install build-essential cmake sudo curl nano git -y

WORKDIR /root
RUN mkdir sdt
WORKDIR /root/sdt

COPY requirements.txt /root/sdt/.

RUN pip install -r requirements.txt
