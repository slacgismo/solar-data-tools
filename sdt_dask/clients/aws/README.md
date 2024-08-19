If you are new to AWS, please follow this step-by-step guide before running the solar-data-tools demo.

# AMI Permissions
You need the appropriate permissions to be able to create and manage ECS clusters. If you don't already have those assigned (perhaps by your IT department), refer to this [setup guide](https://cloudprovider.dask.org/en/latest/aws.html#fargate) for more information.

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

# Docker Image

We recommend using Docker images to run SDT on AWS clusters. For more information on the Docker 
image we provide, see [the Docker README](./docker/README.md).

# AWS Fargate Demo
You can find the AWS Fargate demo notebook in the [examples folder](../../examples). Make sure your AWS
credentials are set correctly as well as any other environment variable that your AWS subscription requires
(e.g. tags). Some common ways to investigate if the cluster initialization fails:
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
3. **Workers failed to run**  
   - Details: the worker logs a report: `Worker is at 85% memory usage`
   - **Solution**: This indicates the cluster might not be powerful enough to handle the workload. Please double-check the worker node specification. Scaling up the cluster, either horizontally or vertically, should resolve the issue.
