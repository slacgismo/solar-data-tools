#!/bin/sh -e

echo "Create the mosek license folder"
cd $HOME
mkdir mosek

echo "Pull the mosek license from s3"
aws s3 cp s3://slac.gismo.ci.artifacts/mosek.license/mosek.lic $HOME/mosek/mosek.lic