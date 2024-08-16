[Under Development]

# Demo notebook
This [notebook](../../examples/tool_demo_azure.ipynb) presents a basic working example of what we'd like to set up with Azure VMs. See below for tips on how to set it up.
## Environment 
- Make sure `dask-cloudprovider[azure]` is installed in your working environment. (This should be included in the full dask-cloudprovider install).
- Add your Azure authentication information to your environment variables. To authenticate a 
user using a password, ensure the variables AZURE_USERNAME and AZURE_PASSWORD are properly set. For more information and for other authentication methods, see this [link](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/identity/azure-identity/TROUBLESHOOTING.md#troubleshoot-environmentcredential-authentication-issues).

## Docker Image
We recommend running a Docker image of your working environment on the Azure VMs. For more information on the Docker 
image we provide, see [the Docker README](./docker/README.md)

## Azure Set Up
You will need to set up a resource group, a virtual network, and a security 
group ahead of creating the cluster. Note that, to allow network traffic to reach your Dask cluster, you will need to create a security group which allows traffic on ports 8786-8787 from wherever you are. For more info, refer to the [Dask docs](https://cloudprovider.dask.org/en/latest/azure.html).