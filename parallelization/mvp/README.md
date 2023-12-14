# MVP Results Archive README

### Overview
This archive contains the results and relevant code from testing the MVP of our solar data solution on Fargate and Local clusters. The tests were run on a Dask FargateCluster, a single threaded LocalCluster and a multi-threaded LocalCluster.

### Contents of the Archive
The archive is organized into two main directories: 'code' and 'results'. Below is a description of each item included:

#### Code
- **`runner.py`**
  - **Overview:** The main script used to execute the MVP on different clusters. This script first retrieves data files from a s3 bucket, processes and analyzes them, and then saves the results to a csv.
  - **Fargate Setup:** To run this script on Fargate, you must ensure that the AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID environment variables are set prior to executing the script. The Fargate cluster will pass these credentials to the containers created to allow them to access s3 resources as needed. More information on the creation of Fargate clusters can be found in the archive documents surrounding Dask.
  - **VM Setup:** To run this script on a VM, first comment out the Fargate cluster object  uncomment the local cluster object in the main function. Then create a virtual machine using your cloud provider of choice. SSH into the VM and install the following dependencies: solar-data-tools, dask, bokeh (to view the dashboasd), and s3fs (for pandas to access s3). See the scripts section below for an example of how to configure an Ubuntu server. Once the necessary dependencies are installed, you must next set the AWS_SECRET_ACCESS_KEY and AWS_ACCESS_KEY_ID environment variables on the VM to allow access to s3. Finally, copy the runner file to the VM and execute it. See the scripts section below for an example scp script to copy the runner.


#### Results
This directory contains both detailed reports and summarized data of the MVP's performance on each cluster.

- **Detailed Dask Reports:** Summary of the performance of a dask computation. The Task Stream tab shows a visual overview of how tasks are executed on workers and the Worker Profiile (Compute) tab shows a flamegraph of the average compute performance for the workers.
  - **`dask-report-fargate.html`**
  - **`dask-report-multi-threaded.html`**
  - **`dask-report-single-threaded.html`**

- **Processing Reports (CSV format):** Verbose output of the DataHandler.runpipeline() function, aggregated over the set of input files and saved to a csv.
  - **`processing_report_fargate.csv`**
  - **`processing_report_multi_threaded.csv`**
  - **`processing_report_single_threaded.csv`**


### VM Setup Notes
- **Setup Script**

      sudo apt-get update
      sudo apt-get install python3-pip python3-venv htop tmux -y
      python3 -m venv .venv
      source .venv/bin/activate
      pip3 install solar-data-tools s3fs bokeh
      python -m pip install "dask[complete]" "dask-cloudprovider[all]"

- **SCP to Copy Runner to AWS VM Home Directory**

      scp -i ~/PATH/TO/PEMFILE.pem ~/PATH/TO/runner.py ubuntu@<VM IP ADDRESS>:~/

