#!/bin/bash

python3 try-parameters.py --input ../dat/dataset-toy.csv --output_dir PARAMETERS --train_window 252 --test_days 5 --seed 1234 --k 10 --elite_list 5,10,15 --offspring_list 20,30,40 --mutants_list 5,10 --gen_list 50,100,150,200  --verbose

