#!/bin/bash
HELP="Usage: bootstrap-dask [OPTIONS]

AWS EMR Bootstrap Action to install and configure:
    solar-data-tools
        dependencies might be deprecated in the future:
            cassandra-driver (Cassandra might be replaced by other components in future)
    Dask
    Jupyter

By default it does the following things:
- Installs miniconda
- Installs dask, distributed, dask-yarn, pyarrow, and s3fs. This list can be
  extended using the --conda-packages flag below.
- Packages this environment for distribution to the workers.
- Installs and starts a jupyter notebook server running on port 8888. This can
  be disabled with the --no-jupyter flag below.

Options:
    --password, -pw             Set the password for the Jupyter Notebook
                                Server. Default is 'dask-user'.
    --conda-packages            Extra packages to install from conda.
"

set -e

rm -rf ./root/miniconda
rm -rf ./sdt

# Parse Inputs. This is specific to this script, and can be ignored
# -----------------------------------------------------------------
JUPYTER_PASSWORD="dask-user"
EXTRA_CONDA_PACKAGES=""
JUPYTER="true"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "$HELP"
            exit 0
            ;;
        -pw|--password)
            JUPYTER_PASSWORD="$2"
            shift
            shift
            ;;
        --conda-packages)
            shift
            PACKAGES=()
            while [[ $# -gt 0 ]]; do
                case $1 in
                    -*)
                        break
                        ;;
                    *)
                        PACKAGES+=($1)
                        shift
                        ;;
                esac
            done
            EXTRA_CONDA_PACKAGES="${PACKAGES[@]}"
            ;;
        *)
            echo "error: unrecognized argument: $1"
            exit 2
            ;;
    esac
done


# -----------------------------------------------------------------------------
# 1. Check if running on the master node. If not, there's nothing do.
# -----------------------------------------------------------------------------
grep -q '"isMaster": true' /mnt/var/lib/info/instance.json \
|| { echo "Not running on master node, nothing to do" && exit 0; }

# -----------------------------------------------------------------------------
# 2. Install Miniconda
# -----------------------------------------------------------------------------
echo "Installing Miniconda"
curl https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda
rm /tmp/miniconda.sh
echo -e '\nexport PATH=$HOME/miniconda/bin:$PATH' >> $HOME/.bashrc
source $HOME/.bashrc
conda update conda -y

# -----------------------------------------------------------------------------
# 3. Install packages to use in packaged environment
#
# We install a few packages by default, and allow users to extend this list
# with a CLI flag:
#
# - dask-yarn >= 0.7.0, for deploying Dask on YARN.
# - pyarrow for working with hdfs, parquet, ORC, etc...
# - s3fs for access to s3
# - conda-pack for packaging the environment for distribution
# - ensure tornado 5, since tornado 6 doesn't work with jupyter-server-proxy
# -----------------------------------------------------------------------------
echo "Installing base packages"
conda install \
-c conda-forge \
-y \
-q \
dask=2022.1 \
dask-yarn=0.9 \
distributed=2022.1 \
pyarrow \
s3fs \
conda-pack \
tornado \
$EXTRA_CONDA_PACKAGES
# conda-develop \
# dask-bigquery \
# dask-ml \
echo "Successfully installed base packages"

# -----------------------------------------------------------------------------
# 4. Pull solar-data-tools codebase
# -----------------------------------------------------------------------------
echo "Pull solar-data-tools codebase"
sudo yum makecache -q
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake curl nano git glibc-devel

mkdir sdt
cd ./sdt
git clone https://github.com/CMU-Fall23-Practicum/solar-data-tools.git
cd ..
echo "Successfully pulled solar-data-tools codebase"

# -----------------------------------------------------------------------------
# 5. Install solar-data-tools dependencies
# -----------------------------------------------------------------------------
echo "Install solar-data-tools dependencies"
# Add repositories
conda config --add channels stanfordcvxgrp
conda config --add channels conda-forge
conda config --add channels mosek
conda config --add channels slacgismo

# Install dependencies (listed in requirements.txt)
echo "These packages will be installed"
cat ./sdt/solar-data-tools/requirements.txt

conda install -y --quiet --file ./sdt/solar-data-tools/requirements.txt
conda install -y solar-data-tools
# conda develop ./sdt/solar-data-tools

echo "Finished install"

# Configure Cassandra
conda install -y cassandra-driver
# Check if folder exists, if not, create it

# [!] We are replacing Cassandra and the config directory
# This section needs to be edited when the new system is published
folder=".aws"
if [ ! -d "$folder" ]; then
    mkdir "$folder"
    echo "Created ~/.aws"
else
    echo "~/.aws already exists"
fi
echo $'54.176.95.208\n' > ~/.aws/cassandra_cluster

echo "solar-data-tools dependencies  successfully initialized."

# -----------------------------------------------------------------------------
# 6. Package the environment to be distributed to worker nodes
# -----------------------------------------------------------------------------
echo "Packaging environment"
conda pack -q -o $HOME/environment.tar.gz
# List all packages in the worker environment
echo "Packages installed in the worker environment:"
conda list

# -----------------------------------------------------------------------------
# 7. Configure Dask
#
# This isn't necessary, but for this particular bootstrap script it will make a
# few things easier:
#
# - Configure the cluster's dashboard link to show the proxied version through
#   jupyter-server-proxy. This allows access to the dashboard with only an ssh
#   tunnel to the notebook.
#
# - Specify the pre-packaged python environment, so users don't have to
#
# - Set the default deploy-mode to local, so the dashboard proxying works
#
# - Specify the location of the native libhdfs library so pyarrow can find it
#   on the workers and the client (if submitting applications).
# ------------------------------------------------------------------------------
echo "Configuring Dask"
mkdir -p $HOME/.config/dask
cat <<EOT >> $HOME/.config/dask/config.yaml
distributed:
  dashboard:
    link: "/proxy/{port}/status"

yarn:
  environment: /home/hadoop/environment.tar.gz
  deploy-mode: local

  worker:
    env:
      ARROW_LIBHDFS_DIR: /usr/lib/hadoop/lib/native/

  client:
    env:
      ARROW_LIBHDFS_DIR: /usr/lib/hadoop/lib/native/
EOT
# Also set ARROW_LIBHDFS_DIR in ~/.bashrc so it's set for the local user
echo -e '\nexport ARROW_LIBHDFS_DIR=/usr/lib/hadoop/lib/native' >> $HOME/.bashrc

# -----------------------------------------------------------------------------
# 8. Install jupyter notebook server and dependencies
#
# Install the following packages:
#
# - notebook: the Jupyter Notebook Server
# - ipywidgets: used to provide an interactive UI for the YarnCluster objects
# - jupyter-server-proxy: used to proxy the dask dashboard through the notebook server
# -----------------------------------------------------------------------------
echo "Installing Jupyter"
conda install \
-c conda-forge \
-y \
-q \
notebook \
ipywidgets \
jupyter-server-proxy


# -----------------------------------------------------------------------------
# 9. List all packages in the client environment
# -----------------------------------------------------------------------------
echo "Packages installed in the client environment:"
conda list


# -----------------------------------------------------------------------------
# 10. Configure Jupyter Notebook
# -----------------------------------------------------------------------------
echo "Configuring Jupyter"
mkdir -p $HOME/.jupyter
HASHED_PASSWORD=`python -c "from jupyter_server.auth import passwd; print(passwd('$JUPYTER_PASSWORD'))"`
cat <<EOF >> $HOME/.jupyter/jupyter_notebook_config.py
c.NotebookApp.password = u'$HASHED_PASSWORD'
c.NotebookApp.open_browser = False
c.NotebookApp.ip = '0.0.0.0'
EOF


# -----------------------------------------------------------------------------
# 11. Define an on-start service for the Jupyter Notebook Server
#
# This sets the notebook server up to properly run as a background service.
# -----------------------------------------------------------------------------
echo "Configuring Jupyter Notebook On-start Service"

sudo bash -c 'echo "[Unit]
Description=Jupyter Notebook Server
After=network.target

[Service]
User=hadoop
Restart=always
ExecStart=/bin/bash -c '\''exec su - hadoop -c \"jupyter notebook >> /var/log/jupyter-notebook.log 2>&1\"'\''

[Install]
WantedBy=multi-user.target" > /etc/systemd/system/jupyter-notebook.service'

# -----------------------------------------------------------------------------
# 11. Start the Jupyter Notebook Server
# -----------------------------------------------------------------------------
echo "Starting Jupyter Notebook Server"
# sudo initctl reload-configuration
# sudo initctl start jupyter-notebook
## initctl is not available in CentOS, try systemctl below
sudo systemctl daemon-reload
sudo systemctl start jupyter-notebook