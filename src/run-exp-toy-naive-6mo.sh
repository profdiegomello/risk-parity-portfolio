#!/bin/bash


#############################################
#
# General Parameters
#
#############################################

# Miscellaneous
SEED=$RANDOM
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")

# Input/output parameters
INPUT_FILE=../dat/dataset-toy-6mo.csv
OUTPUT_FOLDER="./TOY-K-10-Sharpe-25"

# Portfolio parameters
TRANSACTION_COST=0.005
K=10
QUARTILE=0.25

# Backtest parameters
TEST_WINDOW=1
TRAIN_WINDOW=252

# BRKGA parameters
GENERATIONS=150
ELITE=10
OFFSPRING=40
MUTANTS=10
PROB_BIAS=0.7
WORKERS=8
    
#############################################
#
# Naive
#
#############################################
python3 experiment_runner.py \
    --input            $INPUT_FILE \
    --output_dir       $OUTPUT_FOLDER \
    --strategy         naive \
    --solver           SLSQP \
    --k                $K \
    --train_window     $TRAIN_WINDOW \
    --test_window      $TEST_WINDOW \
    --quartile_filter  $QUARTILE \
    --transaction_cost $TRANSACTION_COST \
    --n_elites         $ELITE \
    --n_offsprings     $OFFSPRING \
    --n_mutants        $MUTANTS \
    --bias             $PROB_BIAS \
    --n_gen            $GENERATIONS \
    --workers          $WORKERS \
    --seed             $SEED \
    --verbose

    
echo
echo '======================================================================'
echo
    
#############################################
#
# Post-processing scripts
#
#############################################
python3 plot_report.py --output_dir $OUTPUT_FOLDER --k $K --quartile $QUARTILE
