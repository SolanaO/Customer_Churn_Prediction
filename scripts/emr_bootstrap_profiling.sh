#!/bin/bash

python3 -m venv /home/hadoop/path/to/venv
source /home/hadoop/path/to/venv/bin/activate

set -x

python3 -m pip install --upgrade pip
pip3 freeze
pip3 install Cython==0.29.24
pip3 freeze
yum install python3-devel -y
pip3 freeze
yum install -y libjpeg-devel -v
pip3 freeze
pip3 install numpy==1.21.2 -v
pip3 freeze
pip3 install pandas==1.3.3  -v
pip3 freeze
pip3 install matplotlib==3.4.3
pip3 freeze

PYSPARK_PYTHON=/home/hadoop/path/to/venv/bin/python3
