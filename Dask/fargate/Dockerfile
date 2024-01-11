FROM python:3.10-slim as base

RUN apt-get update -qq
RUN apt-get install build-essential cmake sudo curl nano git -y

WORKDIR /root
RUN mkdir sdt
WORKDIR /root/sdt
RUN git clone https://github.com/CMU-Fall23-Practicum/solar-data-tools.git

# Install dependencies
WORKDIR /root/sdt/solar-data-tools
RUN python3 -m pip install -e .

RUN pip install dask["complete"]
RUN pip install dask-cloudprovider[all]
RUN pip install bokeh
RUN pip install s3fs
