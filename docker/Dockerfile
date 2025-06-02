FROM python:3.12 as base

WORKDIR /root
RUN mkdir sdt
WORKDIR /root/sdt

RUN pip install solar-data-tools[dask]

# For the s3 dataplug
RUN pip install boto3==1.34.131
