# Before we get started
AWS Fargate is a preferable choice compared to AWS EMR. 

One important thing to note: Dask Yarn has not been updated since 2022. We've discovered that `dask-yarn` 0.9 (the latest version) is not compatible with `dask-distributed` package versions 2022.2 and newer, so we need to pin the version of dask-distributed. However, according to the Dask official guide, EMR requires dask-yarn, which can be problematic because building your software on pinned packages is not ideal. Moreover, the absence of new releases for two years indicates that dask-yarn is not well-supported.

Another challenge is the usage of EMR clusters, both in terms of setup and interaction. Setting up an EMR cluster requires numerous configurations. To interact with an EMR cluster, we need to SSH to the primary node, run a Jupyter notebook from there, and open the notebook in a local browser to submit tasks.

The last problem is the lack of usable example bootstrap scripts. Most of the scripts we can find assume Ubuntu, but EMR is based on CentOS, meaning a different set of commands.

# EMR setup
**Make sure your AWS S3 bucket and EMR cluster are in the same region**
1. Copy the [bootstrap.sh](./bootstrap.sh) to a AWS S3 bucket
2. Go to AWS console, search for EMR
3. Click on `Create Cluster`
4. Setup bootstrap action
   1. Go to `Bootstrap Action` section, click on `add action`, then copy paste the bootstrap.sh URI to the box  
   2. In the script command line argument section, you can set the password of the jupyter notebook. For example, `--password sdt` sets the password to `sdt`. The default password is `dask-user`. Please see the comments in [bootstrap.sh](./bootstrap.sh) for more details
5. Choose/create the ssh keys, fill in the key section and make sure you have a local copy of the .pem file. Save this file to:  
   1. Windows:  `%USERPROFILE%\.aws\credentials`
   2. Linux and MacOS: `~/.aws/credentials`
6. Fill in other sections if needed. For example, your organization might enforce tag policies, then you have to fill in the tag section
7. Click `Create Cluster` and wait till the cluster is ready
8. To create another cluster using the same configuration, you can select this cluster, and click `clone cluster`. No need to go through the settings again

# Interact with an EMR cluster
Here we use the jupyter notebook on EMR primary node to submit tasks to an EMR cluster
1. Open a local terminal and run  
    `ssh -L 8080:localhost:<PORT> <REMOTE_USER>@<REMOTE_HOST> -i <KEY_FILE>.pem`  

    SSH to the primary node using this command on your local machine  
    Where:  
    `PORT`: your selected port number  
    `REMOTE_USER`: hadoop (for example)  
    `REMOTE_HOST`: ec2-xx-xxx-xx-xxx.us-east-2.compute.amazonaws.com  
    `KEY_FILE`: the key file selected/created for cluster configuration  
2. Run this command on primary node to start the jupyter notebook  
    `jupyter notebook --no-browser --port=<PORT>`  
    Where:  
    `PORT`: the PORT selected in step 1  
3. Open a browser from your local machine and navigate to
    [http://localhost:8080/](http://localhost:8080/)  
4. Input password (specified during cluster initialization)
5. Upload [demo](./demo.ipynb) to jupyter notebook directory
6. Run demo

# Debugging Tips
1. **Main issue with EMR**: "format_bytes" not found when "import distributed"
   - Make sure we use this combination
     - dask=2022.1
     - dask-yarn=0.9
     - distributed=2022.1
   - dask-yarn 0.9 is not compatible with new releases after 2022.1  

2. Bootstrap action failed when "Configuring Jupyter"
   - Detail: `ModuleNotFoundError: No module named 'notebook.auth'`
   - Solution: in bootstrap.sh, replace notebook.auth with jupyter_server.auth

3. Can't write .service file: permission denied
   - `sudo bash -c ''[bash cmd you want to run]''`

4. initctl: command not found
   - `initctl` is not available in CentOS, try this alternative: `systemctl`

5. Upstart uses some scripts in /etc/init, what if this dir doesn’t exist?
   - Check if there’s a /etc/init.d , that’s the fall back option

6. conda: package not found
   - Check if the package belongs to a private repository. If yes, find the repo and add it to the channels before start installing the packages. 
   - E.g., For `sig-decomp`, `qss` , add: `conda config --add channels stanfordcvxgrp`

7. Many example bootstrap.sh scripts are based on Ubuntu, but AWS EMR is by defualt based on CentOS. How can we configure EMR to run on Ubuntu?
   - Use customized AMI image, for example:  
    `aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/20.04/stable/current/amd64/hvm/ebs-gp2/ami-id`

8.  [EMR Initialization] Terminated with errors: bootstrap.sh exit with error code 127
    - Check the logs first: node/`i-node_id`/bootstrap_actions/stdout.gz

9. Failed to read bootstrap script from AWS S3 upon initialization
   - Look into logs for specific error information
   - Might be a **REGION** problem: S3 bucket and EMR cluster must be in the same region


   
