FROM python:3.10.11 as base

RUN apt-get update -qq
RUN apt-get install build-essential cmake sudo curl nano git -y

WORKDIR /root
RUN mkdir sdt
WORKDIR /root/sdt

RUN pip install dask["complete"]
RUN pip install dask-cloudprovider[all]
RUN pip install bokeh
RUN pip install s3fs
RUN pip install solar-data-tools
