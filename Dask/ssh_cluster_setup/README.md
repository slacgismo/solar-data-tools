## SSH Dask Cluster
In the case of setting up dask clusters not managed through integrated cloud provider solutions, we consider that there are typically 2 scenarios:  

1. Setting up the cluster with virtual machines on the cloud.  

2. Setting up the cluster with client's own machines.

Generally speaking, steps for setting up the machines are similar for both cases. They can be separated into 2 steps:

1. Install requirements on the machines.

2. Setup the cluster through dask python interface.  
We will provide our python code in this folder. 
### More recommendations for setting up SSH cluster on the cloud
If our client choose to bring their own infrastructure, it is up to our client to configure their network suitable for SSH cluster setup.  
In contrast, we have conducted extensive research in the scenarion where our client chooses to setup the machines through a cloud provider. We experimented with EC2 clusters on AWS. Here are our findings:
* If we run the setup script from within the VPC (e.g. on one of the EC2 machines we've prepared), the cluster can be setup successfully.
* If we run the setup script from a machine from outside the VPC (e.g. our local machine), the setup script will run into a complicated IP binding problem. We ourselves have tried to resolve the problem, but failed after days of work. Many attempt have been made in [this stackoverflow post](https://stackoverflow.com/questions/74265724/best-practices-when-deploying-dask-cloudprovider-ec2-cluster-while-allowing-publ). Some looks promising to us, but we have decided that those are too complicated for the purpose of our tutorial. Compared to the troubles a user would go through trying to run the setup script on their local machine, it makes more sense to just run the script on a machine within the VPC instead.
* In conclusion, we recommend **simply run the script from within the VPC**. Of course, as Dask continues to develope, this may be resolved in the future.
* Our example script is also included in this folder.

### Appendix: some debugging tips while using AWS EC2
1. Please make sure your dependencies are up-to-date. To us the problem was pyopenssl. We have created a [requirements.txt](./requirements.txt) file as a **minimal** set of libraries required to have a ssh cluster working. Make sure everything in it is up to date by running `pip install --upgrade -r requirements.txt`.   
2. To us, setting up ssh clusters with pem files is not really working as expected. But switching to password authentication works. So we would recommend that if you are having authorization issues with ssh setup.  
3. To run solar-data-tools run_pipeline, make sure your EC2 instances has a memory greater than 16 GB. (By default, AWS EC2 instances has a memory of 1GB, and cannot be modified after innitializations).  
4. We have a demo script [here](./ssh_cluster_example.py) that was tested to work on AWS EC2 in December 2023, ubuntu image.  
5. Make sure the python environments are the same across your platforms.  
6. Whereever you are setting up your cluster, please make sure you have sufficient firewall rules. Our recommendation is to allow all traffic during debug phase, and narrow it down once you have everything working.