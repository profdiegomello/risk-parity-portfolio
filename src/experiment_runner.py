import argparse
import numpy as np
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from models import RiskBudgetingBRKGA, MaximumSharpeBRKGA, naive_1_k_allocation
import metrics

def run_backtest(args):
    print(f"[INFO] Carregando dados: {args.input}")
    df_retornos = pd.read_csv(args.input, index_col=0, parse_dates=True)
    df_retornos = df_retornos.dropna(how='all')
    
    # Identificação da coluna de risco zero
    rf_col = 'RISKFREE' if 'RISKFREE' in df_retornos.columns else 'CDI'
    if rf_col not in df_retornos.columns:
        raise ValueError("A série de dados deve conter uma coluna 'RISKFREE' ou 'CDI'.")

    # --- Medidas na Tela (Resumo do Dataset) ---
    universo_ativos = [c for c in df_retornos.columns if c != rf_col]
    n_ativos = len(universo_ativos)
    n_linhas = len(df_retornos)
    data_min = df_retornos.index.min().strftime('%d/%m/%Y')
    data_max = df_retornos.index.max().strftime('%d/%m/%Y')

    print(f"============================================================")
    print(f"ESTATÍSTICAS DO DATASET")
    print(f"------------------------------------------------------------")
    print(f"Total de Ativos (N) : {n_ativos}")
    print(f"Total de Registros  : {n_linhas} linhas")
    print(f"Janela Temporal     : {data_min} até {data_max}")
    print(f"Ativo de Referência : {rf_col}")
    print(f"============================================================\n")

    data_inicial_backtest = df_retornos.index[0] + relativedelta(months=args.train_window)
    datas_rebalanceamento = pd.date_range(start=data_inicial_backtest, end=df_retornos.index[-1], freq=f'{args.test_window}MS')

    pesos_historicos = np.zeros(len(universo_ativos)) 
    portfolio_out_of_sample = []

    print(f"============================================================")
    print(f"INICIANDO BACKTEST: {args.strategy.upper()} | K={args.k}")
    print(f"============================================================")

    for data_t in datas_rebalanceamento:
        data_inicio_train = data_t - relativedelta(months=args.train_window)
        data_fim_test = data_t + relativedelta(months=args.test_window)
        
        in_sample = df_retornos.loc[data_inicio_train : data_t - pd.Timedelta(days=1)]
        out_sample = df_retornos.loc[data_t : data_fim_test - pd.Timedelta(days=1)]
        
        if out_sample.empty: break

        print(f"[*] Rebalanceamento: {data_t.strftime('%Y-%m-%d')} | Train: {len(in_sample)} dias | Test: {len(out_sample)} dias")

        ativos_validos = [a for a in universo_ativos if in_sample[a].notna().all() and out_sample[a].notna().all()]
        in_sample = in_sample[ativos_validos + [rf_col]]
        out_sample = out_sample[ativos_validos]
        
        rf_dinamico = in_sample[rf_col].mean()
        ret_medios = in_sample[ativos_validos].mean().values
        cov_matrix = in_sample[ativos_validos].cov().values
        
        volatilidade = np.sqrt(np.diag(cov_matrix))
        sharpes_hist = np.where(volatilidade > 0, (ret_medios - rf_dinamico) / volatilidade, -np.inf)
        percentil_corte = np.quantile(sharpes_hist, 1.0 - args.quartile_filter)
        ativos_quartil_idx = np.where(sharpes_hist >= percentil_corte)[0]
        
        if len(ativos_quartil_idx) < args.k:
            print(f"[AVISO] Universo filtrado ({len(ativos_quartil_idx)}) menor que K ({args.k}). Pulando janela.")
            continue
            
        pesos_novos_janela = np.zeros(len(ativos_validos))
        
        if args.strategy == 'naive':
            top_k, p_naive = naive_1_k_allocation(ret_medios[ativos_quartil_idx], cov_matrix[np.ix_(ativos_quartil_idx, ativos_quartil_idx)], rf_dinamico, args.k)
            pesos_novos_janela[ativos_quartil_idx[top_k]] = p_naive[top_k]
            
        else:
            algorithm = BRKGA(n_elites=args.n_elites, n_offsprings=args.n_offsprings, n_mutants=args.n_mutants, bias=args.bias)
            
            if args.strategy in ['rp_convex', 'rp_nonconvex']:
                form = 'convex' if args.strategy == 'rp_convex' else 'non_convex'
                problem = RiskBudgetingBRKGA(cov_matrix[np.ix_(ativos_quartil_idx, ativos_quartil_idx)], args.k, formulation=form, solver_method=args.solver)
            elif args.strategy == 'msr':
                problem = MaximumSharpeBRKGA(ret_medios[ativos_quartil_idx], cov_matrix[np.ix_(ativos_quartil_idx, ativos_quartil_idx)], rf_dinamico, args.k)

            # --- Verbose mode aqui ---
            res = pymoo_minimize(problem, algorithm, ("n_gen", args.n_gen), seed=args.seed, verbose=args.verbose)
            
            indices_selecionados_sub = problem._decode(res.X)
            indices_globais = ativos_quartil_idx[indices_selecionados_sub]
            sub_cov = cov_matrix[np.ix_(indices_globais, indices_globais)]

            if args.strategy == 'rp_convex':
                y0 = np.ones(args.k) / args.k
                res_cont = minimize(problem._obj_convex, y0, args=(sub_cov, problem.b_target), method='L-BFGS-B', bounds=tuple((1e-8, None) for _ in range(args.k)))
                w_final = res_cont.x / np.sum(res_cont.x)
            elif args.strategy == 'rp_nonconvex':
                bounds = tuple((0.0, 1.0) for _ in range(args.k))
                x0 = np.ones(args.k) / args.k
                constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
                res_cont = minimize(problem._obj_non_convex, x0, args=(sub_cov, problem.b_target), method='SLSQP', bounds=bounds, constraints=constraints)
                w_final = res_cont.x / np.sum(res_cont.x)
            elif args.strategy == 'msr':
                sub_ret = ret_medios[indices_globais]
                bounds = tuple((0.0, 1.0) for _ in range(args.k))
                x0 = np.ones(args.k) / args.k
                constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
                res_cont = minimize(problem._neg_sharpe, x0, args=(sub_ret, sub_cov, rf_dinamico), method='SLSQP', bounds=bounds, constraints=constraints)
                w_final = res_cont.x
                
            pesos_novos_janela[indices_globais] = w_final

        pesos_novos_global = np.zeros(len(universo_ativos))
        for idx_local, ativo in enumerate(ativos_validos):
            idx_global = universo_ativos.index(ativo)
            pesos_novos_global[idx_global] = pesos_novos_janela[idx_local]

        turnover = metrics.calculate_turnover(pesos_historicos, pesos_novos_global)
        custo_execucao = turnover * args.transaction_cost
        
        retornos_diarios_oos = out_sample.dot(pesos_novos_janela)
        retornos_diarios_oos.iloc[0] -= custo_execucao
        
        portfolio_out_of_sample.extend(retornos_diarios_oos.values)
        pesos_historicos = pesos_novos_global 

    ts_retornos = pd.Series(portfolio_out_of_sample)
    rf_medio_anual = df_retornos[rf_col].mean() * 252
    
    print(f"\n============================================================")
    print(f"RESULTADOS OUT-OF-SAMPLE ({args.strategy.upper()})")
    print(f"============================================================")
    print(f"Retorno Anualizado : {metrics.annualized_return(ts_retornos):.2%}")
    print(f"Volatilidade Anual : {metrics.annualized_volatility(ts_retornos):.2%}")
    print(f"Índice de Sharpe   : {metrics.sharpe_ratio(ts_retornos, rf_medio_anual):.4f}")
    print(f"Maximum Drawdown   : {metrics.maximum_drawdown(ts_retornos):.2%}")
    print(f"============================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtester: Paridade de Risco vs Benchmarks')
    parser.add_argument('--input', type=str, required=True, help='CSV de retornos.')
    parser.add_argument('--strategy', type=str, choices=['rp_convex', 'rp_nonconvex', 'msr', 'naive'], required=True)
    parser.add_argument('--solver', type=str, choices=['SLSQP', 'DE', 'LBFGSB'], default='LBFGSB')
    parser.add_argument('--verbose', action='store_true', help='Exibe detalhes da otimização em cada janela.')
    
    parser.add_argument('--k', type=int, default=20, help='Restrição de cardinalidade.')
    parser.add_argument('--train_window', type=int, default=12, help='Meses de calibração.')
    parser.add_argument('--test_window', type=int, default=3, help='Meses de projeção.')
    parser.add_argument('--quartile_filter', type=float, choices=[0.25, 0.50, 0.75, 1.0], default=1.0)
    parser.add_argument('--transaction_cost', type=float, default=0.005)
    
    parser.add_argument('--n_elites', type=int, default=20)
    parser.add_argument('--n_offsprings', type=int, default=70)
    parser.add_argument('--n_mutants', type=int, default=10)
    parser.add_argument('--bias', type=float, default=0.7)
    parser.add_argument('--n_gen', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    run_backtest(args)
