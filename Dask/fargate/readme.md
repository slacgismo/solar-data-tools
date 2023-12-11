Please follow this step-by-step guide before running the solar-data-tools demo.

# AMI Permissions
1. Please first refer to this [setup guide](https://cloudprovider.dask.org/en/latest/aws.html#fargate)

2. If private image repository needed, also add ECR permission
   (push/pull minimum permissions, full permissions if need to create new repo)

# Get AWS CLI (Recommended)
[Please see this link for details](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

# AWS Credentials
Here we have 2 options
1. Using AWS CLI commands     
   We are taking *long-term credentials* as an example, for other types of credentials, please refer to this [link](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html)
   1. Open a terminal and run `aws configure`
   2. Fill in the requested information:
      1. AWS Access Key ID
      2. AWS Secret Access Key
      3. Default region name (e.g. `us-west-2`)
      4. Default output format (e.g. `json`)

2. Manual setup
   1. Go to the home directory, create directory `.aws` with a file `credentials` in it. It looks like this:  
      1. Windows:  `%USERPROFILE%\.aws\credentials`
      2. Linux and MacOS: `~/.aws/credentials`
   2. Write the credentials into file `credentials`, it looks like:  
        ```
        [default]
        aws_access_key_id=<_KEY_>
        aws_secret_access_key=<_SECRET_>
        ```
# Prepare Environment
## Required Python Packages
```
dask["complete"]  
dask-cloudprovider[all]  
solar-data-tools
```

If you have additional dependencies, install them as well (both Docker image and local environment setup will need them, which are shown below)

## Docker Image

1. Install your python requirements and all dependencies as steps in your dockerfile.  
   Here's a sample dockerfile with solar-data-tools dependencies and all the packages listed above: [link](./Dockerfile)  
2. Build a docker image
   1. Go to the directory where your Dockerfile is
   2. ```docker build -t <YOUR_IMAGE_NAME> .```
3. Push your docker image to a repository. 2 options here:
   1. Dockerhub (**Highly Recommended**)
      1. ```docker tag <YOUR_IMAGE_NAME>:latest <YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:latest```
      2. ```docker push <YOUR_Dockerhub_ID>/<YOUR_IMAGE_NAME>:latest```
   2. ECR  (**Not Recommended**)  
      **This could lead to unexpected issues, use it only when absolutely necessary**
      1. In your AWS console, search `ECR`
      2. Follow the instructions to create a repository or go to an existing one
      3. On the repository page, click `View push commands` on the up right corner
      4. Follow the popped instructions. 

For more details, please follow this [link](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html)

## Client Environment
This refers to the local environment where users run their python scripts to create a cluster and submit the job. To keep your local machine clean, we recommend using `Anaconda` to create a virtual environment and install all required packages there.

1.  Install [Anaconda](https://www.anaconda.com/download/)
2.  Create a virtual environment  
   ```conda create --name YOUR_ENV_NAME python=PYTHON_VERSION```  
   For example  
   ```conda create --name my_dask_env python=3.10```  
   **Important**: this python version must be the same used in the docker image above! Otherwise the client will fail to submit jobs to the cluster
3.  Activate the conda environment  
   ```conda activate YOUR_ENV_NAME```
4.  Install packages listed above, for example
      ```
      # Add this channel for solar-data-tools
      conda config --add channels slacgismo                 
      conda install dask["complete"]
      conda install dask-cloudprovider[all]
      conda install solar-data-tools
      ```
5.  Install [Jupyter Notebook](https://jupyter.org/install) (Recommended)
       -  VSCode plugin (optional): if you are using VSCode, click the sidebar *Extension*, search for *Jupyter* and install the plugin

# Prepare your data plug
Here's the [link](../../notebooks/runner.ipynb) to a sample dataplug

## Example: data provision using local files  

The idea is to load the data from a CSV file as pandas DataFrame, and use the provided function `DataHandler` to convert dataframes into datahandler, then output tuples `(unique_identifier, data_handler)` with which solar-data-tools will perform the analysis.
```Python
def local_csv_to_dh(file):
    """
    Converts a local CSV file into a solar-data-tools DataHandler.
    Parameters:
    - file: Path to the CSV file.
    Returns:
    - A tuple of the file name and its corresponding DataHandler.
    """
    df = pd.read_csv(file, index_col=0)
    # Convert index from int to datetime object
    df.index = pd.to_datetime(df.index)
    dh = DataHandler(df)
    name = os.path.basename(file)
    return (name, dh)
```
Here is an example of the changes: if you wish to utilize remote data, consider replacing the input with a list of unique identifiers for remote databases; for Cassandra, the input should be `siteid`.



Now, let's try the demo!
# Run Demo
1. Get the demo script [here](./demo.ipynb)
2. Open Jupyter Notebook, set the python interpreter (in the drop down list, choose the environment we just created)
3. **Please add your stuff before using this script (see comments for details)**
4. Run the script and wait for cluster initialization, you can check the scheduler and workers status via AWS ECS console
5. If it does not work, please check:
   1. Is the cluster created? In ECS console, click cluster id
   2. Is the scheduler running? In cluster console, click the tab 'tasks' to find the scheduler
   3. Want detailed error messages? Click on each worker/scheduler and then the **logs** tab for specific information


# Debugging Tips
1. **Error during deserialization of the task graph**   
     - **Solution**:
     Please make sure the client environment aligns with the scheduler environment!  (e.g. python versions, python packages & versions) 

2. **Windows Powershell compatibility issues with AWS** 
   - Details: docker push command shows `retrying in ... second ... EOF`
   - **Solution**: Use Git Bash on windows can save you a lot of time.
3. **Scheduler failed to pull the image**
   -  Details: sometimes it reports "unable to retreive ecr registry auth ... please check your task network configuration"
   - **Solution**: If your image is in a *ECR* repository (private or public), this is the hazardous situation. Consider switching to *Dockerhub*, which has been proven to be perfectly compatible with Fargate.  
4. **Workers failed to run**  
   - Details: the worker logs a report: `Worker is at 85% memory usage`
   - **Solution**: This indicates the cluster might not be powerful enough to handle the workload. Please double-check the worker node specification. Scaling up the cluster, either horizontally or vertically, should resolve the issue.
