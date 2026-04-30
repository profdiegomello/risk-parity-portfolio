#!/bin/bash


#############################################
#
# General Parameters
#
#############################################

# Miscellaneous
SEED=$RANDOM
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")

# Data treatment
OUTLIER_METHOD="winsor"
WINSOR_LIMITS=0.01
IQR_MULT=3.0 

# Input/output parameters
INPUT_FILE=../dat/dataset-3.csv
OUTPUT_FOLDER="./LAPO-K-20-Sharpe-25-Dataset-3"

# Portfolio parameters
TRANSACTION_COST=0.0
K=20
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
# Convex
#
#############################################
python3 experiment_runner.py \
    --input            $INPUT_FILE \
    --output_dir       $OUTPUT_FOLDER \
    --strategy         rp_convex \
    --solver           SLSQP \
    --k                $K \
    --train_window     $TRAIN_WINDOW \
    --test_window      $TEST_WINDOW \
    --quartile_filter  $QUARTILE \
    --transaction_cost $TRANSACTION_COST \
    --outlier_method   $OUTLIER_METHOD \
    --winsor_limits    $WINSOR_LIMITS \
    --iqr_multiplier   $IQR_MULT \
    --n_elites         $ELITE \
    --n_offsprings     $OFFSPRING \
    --n_mutants        $MUTANTS \
    --bias             $PROB_BIAS \
    --n_gen            $GENERATIONS \
    --workers          $WORKERS \
    --seed             $SEED \
    --verbose \
    --warmstart

echo
echo '======================================================================'
echo

#############################################
#
# Post-processing scripts
#
#############################################
python3 plot_report.py --output_dir $OUTPUT_FOLDER --k $K --quartile $QUARTILE
