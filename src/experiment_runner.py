import argparse
import numpy as np
import pandas as pd
import os
import multiprocessing
import time  # Para medição de performance
from scipy.optimize import minimize
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from models import RiskBudgetingBRKGA, MaximumSharpeBRKGA, MinimumVarianceBRKGA, naive_1_k_allocation
import metrics

def run_backtest(args):
    start_total = time.time() # Início da execução global
    
    print(f"[INFO] Carregando dados: {args.input}")
    df_retornos = pd.read_csv(args.input, index_col=0, parse_dates=True)
    df_retornos = df_retornos.dropna(how='all')
    
    rf_col = 'RISKFREE' if 'RISKFREE' in df_retornos.columns else 'CDI'
    if rf_col not in df_retornos.columns:
        raise ValueError("A série de dados deve conter uma coluna 'RISKFREE' ou 'CDI'.")

    universo_ativos = [c for c in df_retornos.columns if c != rf_col]
    n_ativos = len(universo_ativos)
    n_linhas = len(df_retornos)

    print(f"============================================================")
    print(f"ESTATÍSTICAS DO DATASET")
    print(f"------------------------------------------------------------")
    print(f"Total de Ativos (N) : {n_ativos}")
    print(f"Total de Registros  : {n_linhas} dias úteis")
    print(f"Ativo de Referência : {rf_col}")
    print(f"============================================================\n")

    pesos_historicos = np.zeros(len(universo_ativos)) 
    portfolio_out_of_sample = []
    datas_oos_globais = []
    log_pesos_diarios = [] 
    log_tempos_rebalanceamento = [] # Lista para armazenar o tempo de cada janela

    # Configuração do Pool de Workers
    n_cores = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_cores) if n_cores > 1 else None
    problem_kwargs = {}
    if pool:
        problem_kwargs['runner'] = pool.starmap
        print(f"[INFO] Processamento paralelo ativado: {n_cores} workers.")

    print(f"============================================================")
    print(f"INICIANDO BACKTEST: {args.strategy.upper()}")
    print(f"============================================================")

    total_dias = len(df_retornos)
    passo_teste = args.test_window
    tamanho_treino = args.train_window

    for idx_atual in range(tamanho_treino, total_dias, passo_teste):
        start_rebal = time.time() # Início da medição da janela
        
        in_sample = df_retornos.iloc[idx_atual - tamanho_treino : idx_atual]
        out_sample = df_retornos.iloc[idx_atual : min(idx_atual + passo_teste, total_dias)]
        
        if out_sample.empty: break

        data_rebalanceamento = out_sample.index[0]

        vols_in = in_sample[universo_ativos].std()
        ativos_validos = vols_in[vols_in > 1e-6].index.intersection(out_sample.columns).tolist()
        
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
            continue
            
        pesos_novos_janela = np.zeros(len(ativos_validos))
        
        # Etapa de Otimização (Gargalo Computacional)
        if args.strategy == 'naive':
            top_k, p_naive = naive_1_k_allocation(ret_medios[ativos_quartil_idx], cov_matrix[np.ix_(ativos_quartil_idx, ativos_quartil_idx)], rf_dinamico, args.k)
            pesos_novos_janela[ativos_quartil_idx[top_k]] = p_naive[top_k]
        else:
            algorithm = BRKGA(n_elites=args.n_elites, n_offsprings=args.n_offsprings, n_mutants=args.n_mutants, bias=args.bias)
            
            if args.strategy in ['rp_convex', 'rp_nonconvex']:
                form = 'convex' if args.strategy == 'rp_convex' else 'non_convex'
                problem = RiskBudgetingBRKGA(cov_matrix[np.ix_(ativos_quartil_idx, ativos_quartil_idx)], args.k, formulation=form, solver_method=args.solver, seed=args.seed, **problem_kwargs)
            elif args.strategy == 'msr':
                problem = MaximumSharpeBRKGA(ret_medios[ativos_quartil_idx], cov_matrix[np.ix_(ativos_quartil_idx, ativos_quartil_idx)], rf_dinamico, args.k, **problem_kwargs)
            elif args.strategy == 'gmv':
                problem = MinimumVarianceBRKGA(cov_matrix[np.ix_(ativos_quartil_idx, ativos_quartil_idx)], args.k, **problem_kwargs)

            res = pymoo_minimize(problem, algorithm, ("n_gen", args.n_gen), seed=args.seed, verbose=args.verbose)
            
            indices_sel = problem._decode(res.X)
            indices_globais = ativos_quartil_idx[indices_sel]
            sub_cov = cov_matrix[np.ix_(indices_globais, indices_globais)]

            # Refinamento Contínuo
            if args.strategy == 'rp_convex':
                y0 = np.ones(args.k) / args.k
                res_cont = minimize(problem._obj_convex, y0, args=(sub_cov, problem.b_target), method=args.solver, bounds=tuple((1e-8, None) for _ in range(args.k)))
                w_final = res_cont.x / np.sum(res_cont.x)
            elif args.strategy == 'rp_nonconvex':
                constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
                res_cont = minimize(problem._obj_non_convex, np.ones(args.k)/args.k, args=(sub_cov, problem.b_target), method='SLSQP', bounds=tuple((0.0, 1.0) for _ in range(args.k)), constraints=constraints)
                w_final = res_cont.x / np.sum(res_cont.x)
            elif args.strategy == 'msr':
                res_cont = minimize(problem._neg_sharpe, np.ones(args.k)/args.k, args=(ret_medios[indices_globais], sub_cov, rf_dinamico), method='SLSQP', bounds=tuple((0.0, 1.0) for _ in range(args.k)), constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
                w_final = res_cont.x
            elif args.strategy == 'gmv':
                res_cont = minimize(problem._obj_variance, np.ones(args.k)/args.k, args=(sub_cov,), method='SLSQP', bounds=tuple((0.0, 1.0) for _ in range(args.k)), constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
                w_final = res_cont.x
                
            pesos_novos_janela[indices_globais] = w_final

        pesos_novos_global = np.zeros(len(universo_ativos))
        for idx_local, ativo in enumerate(ativos_validos):
            idx_global = universo_ativos.index(ativo)
            pesos_novos_global[idx_global] = pesos_novos_janela[idx_local]

        # Turnover e Drift
        turnover = metrics.calculate_turnover(pesos_historicos, pesos_novos_global)
        custo_execucao = turnover * args.transaction_cost
        
        peso_diario_atual = pesos_novos_global.copy()
        
        for data_dia in out_sample.index:
            registro = {'Data': data_dia}
            for i, ativo in enumerate(universo_ativos): registro[ativo] = peso_diario_atual[i]
            log_pesos_diarios.append(registro)
            
            w_validos = np.array([peso_diario_atual[universo_ativos.index(a)] for a in ativos_validos])
            retorno_dia_portfolio = np.dot(w_validos, out_sample.loc[data_dia].values)
            
            if data_dia == out_sample.index[0]: retorno_dia_portfolio -= custo_execucao
                
            portfolio_out_of_sample.append(retorno_dia_portfolio)
            datas_oos_globais.append(data_dia)
            
            w_drifted = w_validos * (1 + out_sample.loc[data_dia].values)
            soma_w = np.sum(w_drifted)
            if soma_w > 0: w_drifted /= soma_w
                
            peso_diario_atual = np.zeros(len(universo_ativos))
            for i, a in enumerate(ativos_validos): peso_diario_atual[universo_ativos.index(a)] = w_drifted[i]

        pesos_historicos = peso_diario_atual 
        
        # Fim da medição da janela
        tempo_janela = time.time() - start_rebal
        log_tempos_rebalanceamento.append({'Data': data_rebalanceamento, 'Tempo_Segundos': tempo_janela})
        print(f"[*] Janela {data_rebalanceamento.strftime('%Y-%m-%d')} concluída em {tempo_janela:.2f}s")

    if pool:
        pool.close()
        pool.join()

    # Cálculo final de performance
    end_total = time.time() - start_total
    ts_retornos = pd.Series(portfolio_out_of_sample, index=datas_oos_globais)
    rf_medio_anual = df_retornos[rf_col].mean() * 252
    
    ret_anual = metrics.annualized_return(ts_retornos)
    vol_anual = metrics.annualized_volatility(ts_retornos)
    sharpe = metrics.sharpe_ratio(ts_retornos, rf_medio_anual)
    sortino = metrics.sortino_ratio(ts_retornos, rf_medio_anual)
    mdd = metrics.maximum_drawdown(ts_retornos)

    print(f"\n============================================================")
    print(f"RESULTADOS FINAIS - TEMPO TOTAL: {end_total:.2f}s")
    print(f"============================================================")
    print(f"Índice de Sharpe   : {sharpe:.4f}")
    print(f"Índice de Sortino  : {sortino:.4f}")
    print(f"============================================================\n")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        id_exp = f"{args.strategy}_K{args.k}_Q{int(args.quartile_filter*100)}"
        
        # 1. Log de Tempos de Execução (Novo)
        pd.DataFrame(log_tempos_rebalanceamento).to_csv(os.path.join(args.output_dir, f"exec_times_{id_exp}.csv"), index=False)
        
        # 2. Retornos e Pesos
        ts_retornos.to_csv(os.path.join(args.output_dir, f"oos_ts_{id_exp}.csv"), header=['Retorno'])
        pd.DataFrame(log_pesos_diarios).set_index('Data').to_csv(os.path.join(args.output_dir, f"pesos_diarios_{id_exp}.csv"))
        
        # 3. Atualizar Tabela Mestre (com Tempo Total)
        arquivo_mestre = os.path.join(args.output_dir, "resultados_mestre.csv")
        df_metricas = pd.DataFrame({
            'Estrategia': [args.strategy], 'K': [args.k], 'Quartil': [args.quartile_filter],
            'Retorno_Anual': [ret_anual], 'Volatilidade_Anual': [vol_anual], 
            'Sharpe': [sharpe], 'Sortino': [sortino], 'MDD': [mdd],
            'Tempo_Total_Seg': [end_total] # Incluído tempo total
        })
        df_metricas.to_csv(arquivo_mestre, mode='a', header=not os.path.exists(arquivo_mestre), index=False)
        
        # 4. Tabela Resumo LaTeX (com Tempo Total)
        txt_path = os.path.join(args.output_dir, "tabela_resumo_latex.txt")
        with open(txt_path, "a") as f:
            if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
                f.write("Modelo\tSharpe\tSortino\tMDD\tTempo_Total(s)\n")
            f.write(f"{args.strategy.upper()}\t{sharpe:.4f}\t{sortino:.4f}\t{mdd:.4f}\t{end_total:.2f}\n")
            
        print(f"[INFO] Logs de execução salvos em: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtester: Risk Parity Performance Log')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--strategy', type=str, required=True, choices=['rp_convex', 'rp_nonconvex', 'msr', 'gmv', 'naive'])
    parser.add_argument('--solver', type=str, default='SLSQP')
    parser.add_argument('--output_dir', type=str)
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
