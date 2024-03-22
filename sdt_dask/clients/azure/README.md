[Under Development]

# Demo notebook
This [notebook](https://github.com/slacgismo/solar-data-tools/blob/d884f73363680d41b2545516fbb2d89bd6de42b5/sdt_dask/examples/tool_demo_azure.ipynb) presents a basic working example of what we'd like to set up with Azure VMs. See below for tips on how to set it up.
## Environment 
- Make sure `dask-cloudprovider[azure]` is installed in your working environment. (This should be included in the full dask-cloudprovider install).
- Add your Azure authentication information to your environment variables. To authenticate a 
user using a password, ensure the variables AZURE_USERNAME and AZURE_PASSWORD are properly set. For more information and for other authentication methods, see this [link](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/identity/azure-identity/TROUBLESHOOTING.md#troubleshoot-environmentcredential-authentication-issues).

## Docker Image
We recommend running a Docker image of your working environment on the Azure VMs. 
1. Install your python requirements and all dependencies as steps in your Dockerfile. The image needs to have the packages needed to run the pipelines (at least: solar-data-tools, dask and dask-cloudprovider, and any others needed for dataplugs, etc). 
2. Build a docker image
   1. Go to the directory where your Dockerfile is
   2. ```docker build -t <YOUR_IMAGE_NAME> .```
3. Push your docker image to a repository.
   1. Dockerhub (**Highly Recommended**)
      1. ```docker tag <YOUR_IMAGE_NAME>:latest <YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:latest```
      2. ```docker push <YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:latest```'

## Azure Set Up
You will need to set up a resource group, a virtual n etwork, and a security 
group ahead of creating the cluster. Note that, to allow network traffic to reach your Dask cluster, you will need to create a security group which allows traffic on ports 8786-8787 from wherever you are. For more info, refer to the [Dask docs](https://cloudprovider.dask.org/en/latest/azure.html).