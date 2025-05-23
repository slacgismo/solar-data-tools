# Docker images for running the SDT Dask Tool

We recommend using Docker images to run on cloud providers (AWS, Azure, GCP, etc). Your local environment
should ideally match the Docker image installed on the Dask scheduler and workers. The main packages that
need to match are `numpy`, `dask`, `dask-cloudprovider`, and `distributed`.

## Using the provided Docker image (recommended)
For your convenience, we provide a pre-built Docker image with the SDT and Dask dependencies installed for you to run the Dask tool out of the box
(provided you don't need any custom dataplugs). This image can be found under `slacgismo/sdt-v1:latest` and has the
optional Dask dependencies installed (using `pip install solar-data-tools[dask]`).

If you create your local environment using the recommended method in the installation guide (`pip install -e ".[dask]"`),
these dependencies should already match the provided `slacgismo/sdt-v1` image. Note that your local environment needs to have Python 3.12 installed.

## Creating your own image

If you need to create your own image (perhaps due to the need for custom
dataplugs and other packages to retrieve data), we provide a sample Dockerfile [here](./Dockerfile)
for you to get started. You'll just need to add a line to install your required packages,
including any development version of Solar Data Tools that includes your dataplug module.

The example below explains step by step on
creating a basic Docker image for the current version of development branch
of your git repo. Note that you'll need a (free) Docker hub account to push the image to so that it's accessible
from the cloud, and you'll still need to make sure that the Python version and the package versions (at least the
main ones mentioned above) installed in your local environment match what's being installed on your image.

### Install your requirements (incl. your SDT dev version)
Adjust the `Dockerfile` in the `docker/` directory to fit your needs (e.g. add any packages required by
any custom dataplug). To install your development branch (e.g. `my_dev_branch`) on your image,
replace the installation line in the `Dockerfile` with:
```bash
pip install solar-data-tools@git+https://github.com/my_username/solar-data-tools@my_dev_branch
```

### Build your image
After adjusting the `Dockerfile`, you can build your image. Make sure you are in the `docker/` directory and run:

```shell
docker build -t <YOUR_IMAGE_NAME> .
```

Add any tags you want to your image:
```shell
docker tag <YOUR_IMAGE_NAME>:tag <YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:tag
```
Then push the image to your Docker hub account:
```shell
docker push <YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:tag
```

### Pass the image name to your desired client

Once on the Docker hub, the Docker image can now be used by plugging the image into the cluster as
demonstrated below.

FargateClient:
```python
client_setup = FargateClient(workers=3,
                             threads=2,
                             memory=16,
                             image="<YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:tag",
                             tags=TAGS,
                             vpc=VPC,
                             region_name=AWS_DEFAULT_REGION,
                             environment=ENVIRONMENT)
client = client_setup.init_client()
```
AzureClient:

```python
client_setup = AzureClient(workers=3,
                           threads=2,
                           memory=16,
                           resource_group=resource_group,
                           vnet=vnet,
                           security_group=security_group,
                           docker_image="<YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:tag",
                           location=location,
                           vm_size=cpu,
                           public_ingress=True,
                           disk_size=30)
client = client_setup.init_client()
```
