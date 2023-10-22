#### GCP setup guide
1. Go to GCP console, in VPC Network -> Firewall, create a new rule, set fields as shown in the following picture:  
![vpc rule](./readme_imgs/VPC.png)
2. copy the [setup script](./gcp/dask.sh) to any google cloud storage bucket
3. run the following commands, change the variables accourdingly.
```
CLUSTER_NAME=dask
REGION=us-central1
SETUP_SCRIPT=gs://dataproc-staging-us-central1-348367693788-c69smbbk/dask.sh

gcloud dataproc clusters create ${CLUSTER_NAME} \
  --region ${REGION} \
  --master-machine-type n1-standard-8 \
  --worker-machine-type n1-standard-8 \
  --image-version 2.1-ubuntu \
  --initialization-actions  ${SETUP_SCRIPT}\
  --metadata dask-runtime=yarn \
  --optional-components JUPYTER \
  --enable-component-gateway
```

#### References:
https://towardsdatascience.com/serverless-distributed-data-pre-processing-using-dask-amazon-ecs-and-python-part-1-a6108c728cc4
https://aws.amazon.com/blogs/machine-learning/machine-learning-on-distributed-dask-using-amazon-sagemaker-and-aws-fargate/
https://cloud.google.com/blog/products/data-analytics/improve-data-science-experience-using-scalable-python-data-processing