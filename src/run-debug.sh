# General Parameters
TRANSACTION_COST=0.05
TEST_WINDOW=3
GENERATIONS=30
K=15
SEED=$RANDOM

# Convex
python3 experiment_runner.py \
    --input ../dat/dataset-0.csv \
    --output_dir ./resultados_sbpo_debug \
    --strategy rp_convex \
    --solver SLSQP \
    --k $K \
    --train_window 12 \
    --test_window $TEST_WINDOW \
    --quartile_filter 0.25 \
    --transaction_cost $TRANSACTION_COST \
    --n_elites 10 \
    --n_offsprings 35 \
    --n_mutants 5 \
    --n_gen $GENERATIONS \
    --seed $SEED \
    --bias 0.7 \
    --verbose

echo
echo '======================================================================'
echo

# Non-Convex
python3 experiment_runner.py \
    --input ../dat/dataset-0.csv \
    --output_dir ./resultados_sbpo_debug \
    --strategy rp_nonconvex \
    --solver SLSQP \
    --k $K \
    --train_window 12 \
    --test_window $TEST_WINDOW \
    --quartile_filter 0.25 \
    --transaction_cost $TRANSACTION_COST \
    --n_elites 10 \
    --n_offsprings 35 \
    --n_mutants 5 \
    --n_gen $GENERATIONS \
    --seed $SEED \
    --bias 0.7 \
    --verbose
    
echo
echo '======================================================================'
echo

    
# Naive
python3 experiment_runner.py \
    --input ../dat/dataset-0.csv \
    --output_dir ./resultados_sbpo_debug \
    --strategy naive \
    --k $K \
    --train_window 12 \
    --test_window $TEST_WINDOW \
    --quartile_filter 0.25 \
    --transaction_cost $TRANSACTION_COST \
    --n_elites 10 \
    --n_offsprings 35 \
    --n_mutants 5 \
    --n_gen $GENERATIONS \
    --seed $SEED \
    --bias 0.7 \
    --verbose
    
echo
echo '======================================================================'
echo

    
# Global Minimum Variance 
python3 experiment_runner.py \
    --input ../dat/dataset-0.csv \
    --output_dir ./resultados_sbpo_debug \
    --strategy gmv \
    --solver SLSQP \
    --k $K \
    --train_window 12 \
    --test_window $TEST_WINDOW \
    --quartile_filter 0.25 \
    --transaction_cost $TRANSACTION_COST \
    --n_elites 10 \
    --n_offsprings 35 \
    --n_mutants 5 \
    --n_gen $GENERATIONS \
    --seed $SEED \
    --bias 0.7 \
    --verbose
    
echo
echo '======================================================================'
echo

    
# Minimum Sharp 
python3 experiment_runner.py \
    --input ../dat/dataset-0.csv \
    --output_dir ./resultados_sbpo_debug \
    --strategy msr \
    --solver SLSQP \
    --k $K \
    --train_window 12 \
    --test_window $TEST_WINDOW \
    --quartile_filter 0.25 \
    --transaction_cost $TRANSACTION_COST \
    --n_elites 10 \
    --n_offsprings 35 \
    --n_mutants 5 \
    --n_gen $GENERATIONS \
    --seed $SEED \
    --bias 0.7 \
    --verbose
    
# Consolida o resultado
python3 plot_report.py --output_dir ./resultados_sbpo_debug --k 10 --quartile 0.25
