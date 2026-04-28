import argparse
import numpy as np
import pandas as pd
import os
import itertools
import time
import warnings
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from models import RiskBudgetingBRKGA, MaximumSharpeBRKGA, MinimumVarianceBRKGA

warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")

def try_parameters():
    parser = argparse.ArgumentParser(description='BRKGA Hyperparameter Exploration - SBPO 2026')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./param_tests')
    parser.add_argument('--train_window', type=int, default=252)
    parser.add_argument('--test_days', type=int, default=5, help="Numero de dias consecutivos para testar")
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    
    parser.add_argument('--elite_list', type=lambda s: [int(item) for item in s.split(',')], required=True)
    parser.add_argument('--offspring_list', type=lambda s: [int(item) for item in s.split(',')], required=True)
    parser.add_argument('--mutants_list', type=lambda s: [int(item) for item in s.split(',')], required=True)
    parser.add_argument('--gen_list', type=lambda s: [int(item) for item in s.split(',')], required=True)
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df_retornos = pd.read_csv(args.input, index_col=0, parse_dates=True).dropna(axis=1, how='any')
    rf_col = 'RISKFREE' if 'RISKFREE' in df_retornos.columns else 'CDI'
    universo_ativos = [c for c in df_retornos.columns if c != rf_col]
    
    total_dias = len(df_retornos)

    param_combinations = list(itertools.product(
        args.elite_list, 
        args.offspring_list, 
        args.mutants_list, 
        args.gen_list
    ))

    print(f"[INFO] Iniciando exploração: {len(param_combinations)} combinações de parâmetros por dia.")
    print(f"[INFO] O teste ocorrerá por {args.test_days} dia(s) consecutivo(s).")

    dias_testados = 0
    caminho_arquivo = os.path.join(args.output_dir, "param_comparison_master.csv")
    
    if os.path.exists(caminho_arquivo):
        os.remove(caminho_arquivo)

    for idx_atual in range(args.train_window, total_dias, 1):
        if dias_testados >= args.test_days:
            break
            
        in_sample = df_retornos.iloc[idx_atual - args.train_window : idx_atual]
        data_ref = df_retornos.index[idx_atual].strftime('%Y-%m-%d')
        
        rf_din = in_sample[rf_col].mean()
        ret_medios = in_sample[universo_ativos].mean().values
        cov_matrix = in_sample[universo_ativos].cov().values

        print(f"\n[================ Dia: {data_ref} ({dias_testados + 1}/{args.test_days}) ================]")

        for elite, offspring, mutant, gen in param_combinations:
            start_comb = time.time()
            
            algorithm = BRKGA(n_elites=elite, n_offsprings=offspring, n_mutants=mutant, bias=0.7)
            
            problemas = {
                'rp_convex': RiskBudgetingBRKGA(cov_matrix, args.k, formulation='convex', seed=args.seed),
                'rp_nonconvex': RiskBudgetingBRKGA(cov_matrix, args.k, formulation='non_convex', seed=args.seed),
                'msr': MaximumSharpeBRKGA(ret_medios, cov_matrix, rf_din, args.k),
                'gmv': MinimumVarianceBRKGA(cov_matrix, args.k)
            }

            row_result = {'Data': data_ref}

            for name, prob in problemas.items():
                if args.verbose:
                    print(f"---> Otimizando {name.upper()} | E:{elite} O:{offspring} M:{mutant} G:{gen}")
                res = pymoo_minimize(prob, algorithm, ("n_gen", gen), seed=args.seed, verbose=args.verbose)
                row_result[name] = res.F[0] 

            row_result['n_elites'] = elite
            row_result['n_offsprings'] = offspring
            row_result['n_mutants'] = mutant
            row_result['n_gen'] = gen
            row_result['Time_Sec'] = time.time() - start_comb

            # Gravação imediata linha a linha na planilha única
            df_row = pd.DataFrame([row_result])
            df_row.to_csv(caminho_arquivo, mode='a', header=not os.path.exists(caminho_arquivo), index=False)

            if not args.verbose:
                print(f"[*] E:{elite} O:{offspring} M:{mutant} G:{gen} | Registrado no CSV.")

        dias_testados += 1

    print(f"\n[SUCESSO] Exploração concluída. Master salvo em: {caminho_arquivo}")

if __name__ == '__main__':
    try_parameters()
