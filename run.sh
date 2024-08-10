#!/bin/bash

set -ex

python_env=/home/nath/Documents/python_environments/nath/bin/python

#       seed: random state to use for data generations
#          n: size of sample to generate
#     method: logit or probit (only when using binary_choice_models.py)
#    verbose: run in verbose mode
# l1_penalty: l1 penalty to use
# l2_penalty: l2 penalty to use

$python_env binary_choice_models.py --seed 123 -n 1000 --method probit --verbose
$python_env ols.py --seed 42 -n 250 --verbose --gd
$python_env ols.py --seed 42 -n 250 --verbose --l2_penalty 0.1
