# EMR Setup Guide
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

# Interact with a EMR cluster
Here we use the jupyter notebook on EMR primary node to submit tasks to cluster.
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
4. In the browser, input password (specified during cluster initialization)
   
