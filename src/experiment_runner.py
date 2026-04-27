import argparse
import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing.pool import ThreadPool
import time 
from scipy.optimize import minimize
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from models import RiskBudgetingBRKGA, MaximumSharpeBRKGA, MinimumVarianceBRKGA, naive_1_k_allocation
import metrics

def run_backtest(args):
    start_total = time.time()
    
    print(f"[INFO] Carregando dados: {args.input}")
    df_retornos = pd.read_csv(args.input, index_col=0, parse_dates=True)
    df_retornos = df_retornos.dropna(how='all')
    
    rf_col = 'RISKFREE' if 'RISKFREE' in df_retornos.columns else 'CDI'
    if rf_col not in df_retornos.columns:
        raise ValueError("O dataset deve conter uma coluna 'RISKFREE' ou 'CDI'.")

    universo_ativos_bruto = [c for c in df_retornos.columns if c != rf_col]

    # =========================================================================
    # DEFINIÇÃO DO UNIVERSO RESTRITO ESTÁTICO (FILTRO SHARPE GLOBAL)
    # =========================================================================
    in_sample_global = df_retornos.iloc[0 : args.train_window]
    vols_global = in_sample_global[universo_ativos_bruto].std()
    ativos_validos_global = vols_global[vols_global > 1e-6].index.tolist()

    in_sample_global_f = in_sample_global[ativos_validos_global + [rf_col]]
    rf_dinamico_global = in_sample_global_f[rf_col].mean()
    ret_medios_global = in_sample_global_f[ativos_validos_global].mean().values
    cov_matrix_global = in_sample_global_f[ativos_validos_global].cov().values

    volatilidade_global = np.sqrt(np.diag(cov_matrix_global))
    sharpes_hist_global = np.where(volatilidade_global > 0, (ret_medios_global - rf_dinamico_global) / volatilidade_global, -np.inf)
    percentil_corte_global = np.quantile(sharpes_hist_global, 1.0 - args.quartile_filter)
    
    indices_restritos = np.where(sharpes_hist_global >= percentil_corte_global)[0]
    universo_restrito = [ativos_validos_global[i] for i in indices_restritos]
    
    print(f"============================================================")
    print(f"FILTRO GLOBAL ESTÁTICO (Top {int(args.quartile_filter*100)}% Sharpe)")
    print(f"Ativos Brutos: {len(universo_ativos_bruto)} | Ativos no Universo Restrito: {len(universo_restrito)}")
    print(f"============================================================\n")

    # Redução da instância para o universo restrito
    df_retornos = df_retornos[universo_restrito + [rf_col]]
    universo_ativos = universo_restrito

    # =========================================================================
    
    pesos_historicos = np.zeros(len(universo_ativos)) 
    portfolio_out_of_sample = []
    datas_oos_globais = []
    log_pesos_diarios = [] 
    log_tempos_rebalanceamento = [] 
    log_custos_diarios = [] 

    n_cores = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    pool = ThreadPool(n_cores) if n_cores > 1 else None
    problem_kwargs = {}
    if pool:
        problem_kwargs['runner'] = pool.starmap

    total_dias = len(df_retornos)
    passo_teste = args.test_window
    tamanho_treino = args.train_window

    id_exp = f"{args.strategy}_K{args.k}_Q{int(args.quartile_filter*100)}"
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for idx_atual in range(tamanho_treino, total_dias, passo_teste):
        start_rebal = time.time()
        
        in_sample = df_retornos.iloc[idx_atual - tamanho_treino : idx_atual]
        out_sample = df_retornos.iloc[idx_atual : min(idx_atual + passo_teste, total_dias)]
        
        if out_sample.empty: break

        data_rebalanceamento = out_sample.index[0]
        
        # Filtro exclusivo de sobrevivência (evitar variância zero na matriz de covariância)
        vols_in = in_sample[universo_ativos].std()
        ativos_validos = vols_in[vols_in > 1e-6].index.intersection(out_sample.columns).tolist()
        
        if len(ativos_validos) < args.k:
            print(f"[AVISO] Ativos válidos ({len(ativos_validos)}) insuficientes para K={args.k} na data {data_rebalanceamento.date()}. Pulando...")
            continue

        out_sample = out_sample[ativos_validos]
        
        in_sample_f = in_sample[ativos_validos + [rf_col]]
        rf_dinamico = in_sample_f[rf_col].mean()
        ret_medios = in_sample_f[ativos_validos].mean().values
        cov_matrix = in_sample_f[ativos_validos].cov().values
            
        pesos_novos_janela = np.zeros(len(ativos_validos))
        
        # Etapa de Otimização operando sobre o subset estrito validado
        if args.strategy == 'naive':
            top_k, p_naive = naive_1_k_allocation(ret_medios, cov_matrix, rf_dinamico, args.k)
            pesos_novos_janela[top_k] = p_naive[top_k]
        else:
            algorithm = BRKGA(n_elites=args.n_elites, n_offsprings=args.n_offsprings, 
                               n_mutants=args.n_mutants, bias=args.bias)
            
            if args.strategy in ['rp_convex', 'rp_nonconvex']:
                form = 'convex' if args.strategy == 'rp_convex' else 'non_convex'
                problem = RiskBudgetingBRKGA(cov_matrix, args.k, formulation=form, solver_method=args.solver, 
                                            seed=args.seed, **problem_kwargs)
            elif args.strategy == 'msr':
                problem = MaximumSharpeBRKGA(ret_medios, cov_matrix, rf_dinamico, args.k, **problem_kwargs)
            elif args.strategy == 'gmv':
                problem = MinimumVarianceBRKGA(cov_matrix, args.k, **problem_kwargs)

            res = pymoo_minimize(problem, algorithm, ("n_gen", args.n_gen), seed=args.seed, verbose=args.verbose)
            
            indices_sel = problem._decode(res.X)
            sub_cov = cov_matrix[np.ix_(indices_sel, indices_sel)]

            if args.strategy == 'rp_convex':
                res_c = minimize(problem._obj_convex, np.ones(args.k)/args.k, args=(sub_cov, problem.b_target), 
                                 method=args.solver, bounds=tuple((1e-8, None) for _ in range(args.k)))
                w_final = res_c.x / np.sum(res_c.x)
            elif args.strategy == 'rp_nonconvex':
                res_c = minimize(problem._obj_non_convex, np.ones(args.k)/args.k, args=(sub_cov, problem.b_target), 
                                 method='SLSQP', bounds=tuple((0.0, 1.0) for _ in range(args.k)), 
                                 constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
                w_final = res_c.x / np.sum(res_c.x)
            elif args.strategy == 'msr':
                res_c = minimize(problem._neg_sharpe, np.ones(args.k)/args.k, 
                                 args=(ret_medios[indices_sel], sub_cov, rf_dinamico), 
                                 method='SLSQP', bounds=tuple((0.0, 1.0) for _ in range(args.k)), 
                                 constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
                w_final = res_c.x
            elif args.strategy == 'gmv':
                res_c = minimize(problem._obj_variance, np.ones(args.k)/args.k, args=(sub_cov,), 
                                 method='SLSQP', bounds=tuple((0.0, 1.0) for _ in range(args.k)), 
                                 constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
                w_final = res_c.x
            
            pesos_novos_janela[indices_sel] = w_final

        tempo_janela = time.time() - start_rebal
        log_tempos_rebalanceamento.append({'Data': data_rebalanceamento, 'Tempo_Segundos': tempo_janela})

        pesos_novos_global = np.zeros(len(universo_ativos))
        for idx_local, ativo in enumerate(ativos_validos):
            pesos_novos_global[universo_ativos.index(ativo)] = pesos_novos_janela[idx_local]

        turnover = metrics.calculate_turnover(pesos_historicos, pesos_novos_global)
        custo_rebal = turnover * args.transaction_cost
        
        peso_corrente = pesos_novos_global.copy()
        for i_dia, data_dia in enumerate(out_sample.index):
            log_pesos_diarios.append({'Data': data_dia, **{a: peso_corrente[i] for i, a in enumerate(universo_ativos)}})
            
            w_v = np.array([peso_corrente[universo_ativos.index(a)] for a in ativos_validos])
            ret_bruto = np.dot(w_v, out_sample.loc[data_dia].values)
            
            custo_aplicado = custo_rebal if i_dia == 0 else 0.0
            log_custos_diarios.append({'Data': data_dia, 'Custo_Transacao': custo_aplicado})
            
            portfolio_out_of_sample.append(ret_bruto - custo_aplicado)
            datas_oos_globais.append(data_dia)
            
            w_drift = w_v * (1 + out_sample.loc[data_dia].values)
            soma_w = np.sum(w_drift)
            if soma_w > 1e-10: w_drift /= soma_w
            peso_corrente = np.zeros(len(universo_ativos))
            for idx_v, a in enumerate(ativos_validos): peso_corrente[universo_ativos.index(a)] = w_drift[idx_v]

        pesos_historicos = peso_corrente 
        
        print(f"[*] [{args.strategy.upper()}] Rebal: {data_rebalanceamento.strftime('%Y-%m-%d')} "
              f"| Universo Válido: {len(ativos_validos)} ativos -> K={args.k} "
              f"| Tempo: {tempo_janela:.2f}s")

        if args.output_dir:
            pd.Series(portfolio_out_of_sample, index=datas_oos_globais).to_csv(
                os.path.join(args.output_dir, f"oos_ts_{id_exp}.csv"), header=['Retorno']
            )
            pd.DataFrame(log_tempos_rebalanceamento).to_csv(
                os.path.join(args.output_dir, f"exec_times_{id_exp}.csv"), index=False
            )

    if pool:
        pool.close()
        pool.join()

    total_exec = time.time() - start_total
    ts_final = pd.Series(portfolio_out_of_sample, index=datas_oos_globais)
    rf_anual = df_retornos[rf_col].mean() * 252
    
    if args.output_dir:
        df_res = pd.DataFrame({
            'Estrategia': [args.strategy], 'K': [args.k], 'Quartil': [args.quartile_filter],
            'Retorno_Anual': [metrics.annualized_return(ts_final)],
            'Vol_Anual': [metrics.annualized_volatility(ts_final)],
            'Sharpe': [metrics.sharpe_ratio(ts_final, rf_anual)], 
            'Sortino': [metrics.sortino_ratio(ts_final, rf_anual)], 
            'MDD': [metrics.maximum_drawdown(ts_final)],
            'Tempo_Total_Seg': [total_exec]
        })
        master_path = os.path.join(args.output_dir, "resultados_mestre.csv")
        df_res.to_csv(master_path, mode='a', header=not os.path.exists(master_path), index=False)
        
        pd.DataFrame(log_pesos_diarios).set_index('Data').to_csv(os.path.join(args.output_dir, f"pesos_diarios_{id_exp}.csv"))
        pd.DataFrame(log_custos_diarios).to_csv(os.path.join(args.output_dir, f"costs_daily_{id_exp}.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest Solver - SBPO 2026')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--strategy', type=str, required=True, choices=['rp_convex', 'rp_nonconvex', 'msr', 'gmv', 'naive'])
    parser.add_argument('--solver', type=str, default='SLSQP')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--train_window', type=int, default=252)
    parser.add_argument('--test_window', type=int, default=63)
    parser.add_argument('--quartile_filter', type=float, default=1.0)
    parser.add_argument('--transaction_cost', type=float, default=0.005)
    parser.add_argument('--workers', type=int, default=-1)
    parser.add_argument('--n_gen', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_elites', type=int, default=20)
    parser.add_argument('--n_offsprings', type=int, default=70)
    parser.add_argument('--n_mutants', type=int, default=10)
    parser.add_argument('--bias', type=float, default=0.7)
    parser.add_argument('--verbose', action='store_true')
    
    run_backtest(parser.parse_args())
