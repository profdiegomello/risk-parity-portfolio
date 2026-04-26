#!/bin/bash


#############################################
#
# General Parameters
#
#############################################

# Input/output parameters
INPUT_FILE=../dat/dataset-0.csv

# Portfolio parameters
TRANSACTION_COST=0.005
K=15
QUARTILE=0.25

# Backtest parameters
TEST_WINDOW=3
TRAIN_WINDOW=12

# BRKGA parameters
GENERATIONS=30
ELITE=10
OFFSPRING=35
MUTANTS=5
PROB_BIAS=0.7

# Miscellaneous
SEED=$RANDOM
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
OUTPUT_FOLDER="./$TIMESTAMP-SBPO-2026"

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
    --n_elites         $ELITE \
    --n_offsprings     $OFFSPRING \
    --n_mutants        $MUTANTS \
    --bias             $PROB_BIAS \
    --n_gen            $GENERATIONS \
    --seed             $SEED \
    --verbose

echo
echo '======================================================================'
echo

#############################################
#
# Non-Convex
#
#############################################
python3 experiment_runner.py \
    --input            $INPUT_FILE \
    --output_dir       $OUTPUT_FOLDER \
    --strategy         rp_nonconvex \
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
    --seed             $SEED \
    --verbose
    
echo
echo '======================================================================'
echo

    
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
    --seed             $SEED \
    --verbose
    
echo
echo '======================================================================'
echo

    
#############################################
#
# Global Minimum Variance 
#
#############################################
python3 experiment_runner.py \
    --input            $INPUT_FILE \
    --output_dir       $OUTPUT_FOLDER \
    --strategy         gmv \
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
    --seed             $SEED \
    --verbose

    
echo
echo '======================================================================'
echo

    
#############################################
#
# Minimum Sharp 
#
#############################################
python3 experiment_runner.py \
    --input            $INPUT_FILE \
    --output_dir       $OUTPUT_FOLDER \
    --strategy         msr \
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
    --seed             $SEED \
    --verbose

    
#############################################
#
# Post-processing scripts
#
#############################################
python3 plot_report.py --output_dir $OUTPUT_FOLDER --k $K --quartile $QUARTILE
