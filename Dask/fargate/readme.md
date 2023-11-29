** This is just a DRAFT **
To-do list:  
[  ] Replace "image = jjcrush/dask:latest" in demo.ipynb with some other repo name


# AMI Permissions
1. Please first refer to this [setup guide](https://cloudprovider.dask.org/en/latest/aws.html#fargate)

2. If private image repository needed, also add ECR permission
   (push/pull minimum permissions, full permissions if need to create new repo)

# Get AWS CLI (Recommended)
[Please see this link for details](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

# AWS Credentials (Add specific instructions for Windows/Linux)
Here we have 2 options
1. Using AWS CLI commands     
   We are taking *long-term credentials* as an example, for other types of credentials, please refer to this [link](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html))
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
## Python Packages
dask["complete"]  
dask-cloudprovider[all]  
solar-data-tools

If you have additional dependencies, install them as well

## Docker Image
Install your python requirements and all dependencies as steps in your dockerfile.


(**To be uploaded**)  
There's a A sample docker image with solar-data-tools dependencies and all the packages listed above. Please find the 

** Tell users how to find the "view push commands" page **

** Give them example dockerfile. **

** Please use dockerhub to store your image and make it public (saves you a lot of time to debug) **



To create your own image, make sure you have all the packages included and follow this [link](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html)

# Run Demo
** Explain: how to run the demo **
** Be specific: e.g., create a repo in us-east-2 or ```YOUR_REGION```, name: dask **


# Issues
1. **RuntimeError**: Error during deserialization of the task graph  
   - **[Solution]**
   Please make sure the client environment aligns with the scheduler environment! e.g. python versions, python packages & versions. 

2. **Windows Powershell compatibility issues with AWS**:  

3. ** Scheduler failed to pull the image **
   ** Docker Image Repository **
