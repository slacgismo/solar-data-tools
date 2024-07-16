FROM python:3.10 as base

WORKDIR /root
RUN mkdir sdt
WORKDIR /root/sdt

COPY ../../requirements.txt /root/sdt/.

RUN pip install -r requirements.txt
