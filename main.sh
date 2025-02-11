#!/bin/bash

# nohup VERSION=191 bash main.sh > /dev/null 2> /dev/null &

# Exit on any failure.
set -e

ctrlc() {
    killall python
    mn -c
    exit
}

trap ctrlc INT

# Step 1. Train the multi-task prediction model
python train.py --data_path "KuaiRand-1K" --split_date "2022-05-05"

# Step 2. Predict multi-task scores for each user item pair
python test.py --data_path "KuaiRand-1K" --start_date "2022-05-06" --end_date "2022-05-06" --per_select_num 1

# Step 3. Sample user item pairs
python gen_sample.py --data_path "KuaiRand-1K" --split_date "2022-05-05" --user_sample_size 10000

# Step 4. Complie cpp code of IBR
g++ --std=c++11 -shared -fPIC -o multi_bisect.so multi_bisect.cpp

# Step 5. Do experiments (gen plots)
python gen_plot.py --plot=1
python gen_plot.py --plot=2
python gen_plot.py --plot=3
python gen_plot.py --plot=4
python gen_plot.py --plot=5
python gen_plot.py --plot=6
python gen_plot.py --plot=7
