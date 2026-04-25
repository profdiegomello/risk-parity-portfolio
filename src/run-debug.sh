# Convex
python3 experiment_runner.py \
    --input ../dat/dataset-0.csv \
    --strategy rp_convex \
    --solver SLSQP \
    --k 10 \
    --train_window 12 \
    --test_window 6 \
    --quartile_filter 0.25 \
    --transaction_cost 0.0 \
    --n_elites 10 \
    --n_offsprings 35 \
    --n_mutants 5 \
    --n_gen 10 \
    --seed 12345 \
    --bias 0.7 \
    --verbose

# Non-Convex
python3 experiment_runner.py \
    --input ../dat/dataset-0.csv \
    --strategy rp_nonconvex \
    --solver SLSQP \
    --k 10 \
    --train_window 12 \
    --test_window 6 \
    --quartile_filter 0.25 \
    --transaction_cost 0.0 \
    --n_elites 10 \
    --n_offsprings 35 \
    --n_mutants 5 \
    --n_gen 10 \
    --seed 12345 \
    --bias 0.7 \
    --verbose
    
# Naive
python3 experiment_runner.py \
    --input ../dat/dataset-0.csv \
    --strategy naive \
    --k 10 \
    --train_window 12 \
    --test_window 6 \
    --quartile_filter 0.25 \
    --transaction_cost 0.0 \
    --n_elites 10 \
    --n_offsprings 35 \
    --n_mutants 5 \
    --n_gen 10 \
    --seed 12345 \
    --bias 0.7 \
    --verbose
    
# Markowitz
python3 experiment_runner.py \
    --input ../dat/dataset-0.csv \
    --strategy msr \
    --solver SLSQP \
    --k 10 \
    --train_window 12 \
    --test_window 6 \
    --quartile_filter 0.25 \
    --transaction_cost 0.0 \
    --n_elites 10 \
    --n_offsprings 35 \
    --n_mutants 5 \
    --n_gen 10 \
    --seed 12345 \
    --bias 0.7 \
    --verbose
