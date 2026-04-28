import argparse
import multiprocessing
import os
import time
import warnings
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize as pymoo_minimize
from scipy.optimize import minimize

import metrics
from models import (
    MaximumSharpeBRKGA,
    MinimumVarianceBRKGA,
    RiskBudgetingBRKGA,
    naive_1_k_allocation,
)

# Suppress numerical warnings from SLSQP near the bounds.
warnings.filterwarnings(
    "ignore",
    message="Values in x were outside bounds during a minimize step, clipping to bounds",
)


def build_dynamic_universe(ret_medios, cov_matrix, rf_dinamico, quartile_filter, k_minimo):
    sharpes_hist = metrics.calcular_vetor_sharpe(ret_medios, cov_matrix, rf_dinamico)
    percentil_corte = np.quantile(sharpes_hist, 1.0 - quartile_filter)
    indices_restritos = np.where(sharpes_hist >= percentil_corte)[0]

    # Guarantee at least K assets in case the quantile cut becomes too restrictive.
    if len(indices_restritos) < k_minimo:
        indices_ordenados = np.argsort(sharpes_hist)[::-1]
        indices_restritos = indices_ordenados[: min(k_minimo, len(indices_ordenados))]

    return indices_restritos, sharpes_hist


def sanitize_returns(in_sample_assets, application_assets, method, winsor_limits, iqr_multiplier):
    in_sample_assets = in_sample_assets.astype(float).copy()
    application_assets = application_assets.astype(float).copy()

    if method == "winsor":
        lower = in_sample_assets.quantile(winsor_limits)
        upper = in_sample_assets.quantile(1.0 - winsor_limits)
    elif method == "iqr":
        q1 = in_sample_assets.quantile(0.25)
        q3 = in_sample_assets.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
    else:
        raise ValueError(f"Metodo de saneamento invalido: {method}")

    lower = lower.fillna(-np.inf)
    upper = upper.fillna(np.inf)

    in_sample_sanitized = in_sample_assets.clip(lower=lower, upper=upper, axis=1)
    application_sanitized = application_assets.clip(lower=lower, upper=upper)

    stats = {
        "n_in_sample_clipped": int((in_sample_sanitized != in_sample_assets).sum().sum()),
        "n_application_clipped": int((application_sanitized != application_assets).sum()),
    }

    return in_sample_sanitized, application_sanitized, stats


def normalized_weights_or_none(weights):
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights)):
        return None

    soma = np.sum(weights)
    if soma <= 1e-10:
        return None

    pesos = weights / soma
    if not np.all(np.isfinite(pesos)):
        return None

    return pesos


def decode_brkga_solution_or_none(problem, solution_vector, expected_k):
    if solution_vector is None:
        return None

    solution_vector = np.asarray(solution_vector, dtype=float)
    if solution_vector.ndim != 1:
        return None

    if len(solution_vector) < expected_k or not np.all(np.isfinite(solution_vector)):
        return None

    indices_sel = problem._decode(solution_vector)
    indices_sel = np.asarray(indices_sel, dtype=int)

    if len(indices_sel) != expected_k:
        return None

    if len(np.unique(indices_sel)) != expected_k:
        return None

    if np.any(indices_sel < 0) or np.any(indices_sel >= problem.n_assets):
        return None

    return indices_sel


def fallback_subset_indices(strategy, ret_medios, cov_matrix, k_efetivo):
    if strategy == "msr":
        ordenacao = np.argsort(ret_medios)[::-1]
        return np.asarray(ordenacao[:k_efetivo], dtype=int)

    if strategy == "gmv":
        n_assets = cov_matrix.shape[0]
        ativos_restantes = list(range(n_assets))
        subconjunto = []

        variancias = np.diag(cov_matrix)
        primeiro = int(np.argmin(variancias))
        subconjunto.append(primeiro)
        ativos_restantes.remove(primeiro)

        while len(subconjunto) < k_efetivo and ativos_restantes:
            melhor_ativo = None
            melhor_variancia = np.inf

            for candidato in ativos_restantes:
                subconjunto_teste = subconjunto + [candidato]
                sub_cov = cov_matrix[np.ix_(subconjunto_teste, subconjunto_teste)]
                pesos_eq = np.ones(len(subconjunto_teste)) / len(subconjunto_teste)
                variancia_portfolio = float(pesos_eq.T @ sub_cov @ pesos_eq)

                if variancia_portfolio < melhor_variancia:
                    melhor_variancia = variancia_portfolio
                    melhor_ativo = candidato

            subconjunto.append(melhor_ativo)
            ativos_restantes.remove(melhor_ativo)

        return np.asarray(subconjunto, dtype=int)

    else:
        volatilidades = np.sqrt(np.clip(np.diag(cov_matrix), a_min=0.0, a_max=None))
        ordenacao = np.argsort(volatilidades)
        return np.asarray(ordenacao[:k_efetivo], dtype=int)


def run_backtest(args):
    start_total = time.time()

    print(f"[INFO] Carregando dados: {args.input}")
    df_retornos = pd.read_csv(args.input, index_col=0, parse_dates=True)

    # 1. Tratamento de dados.
    df_retornos = df_retornos.dropna(axis=0, how="all")
    df_retornos = df_retornos.dropna(axis=1, how="any")

    rf_col = "RISKFREE" if "RISKFREE" in df_retornos.columns else "CDI"
    if rf_col not in df_retornos.columns:
        raise ValueError("O dataset deve conter uma coluna 'RISKFREE' ou 'CDI' sem valores NA.")

    universo_ativos_bruto = [c for c in df_retornos.columns if c != rf_col]

    # 2. Filtro inicial de liquidez/baixa volatilidade.
    vols_totais = df_retornos[universo_ativos_bruto].std()
    ativos_negociados = vols_totais[vols_totais > 1e-6].index.tolist()
    df_retornos = df_retornos[ativos_negociados + [rf_col]]
    universo_ativos = ativos_negociados

    if args.test_window != 1:
        print(
            f"[WARN] test_window={args.test_window} foi solicitado, "
            "mas este runner agora executa rebalanceamento diario com aplicacao em t+1. "
            "O valor sera ignorado."
        )

    id_exp = f"{args.strategy}_K{args.k}_Q{int(args.quartile_filter * 100)}"
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    pesos_historicos = np.zeros(len(universo_ativos))
    portfolio_oos_bruto = []
    portfolio_oos_liquido = []
    datas_oos_globais = []
    log_pesos_diarios = []
    log_tempos_rebalanceamento = []
    log_custos_diarios = []
    log_universos_dinamicos = []
    log_sanitizacao = []
    log_otimizacao = []

    n_cores = args.workers if args.workers > 0 else multiprocessing.cpu_count()
    pool = ThreadPool(n_cores) if n_cores > 1 else None
    problem_kwargs = {}
    if pool:
        problem_kwargs["runner"] = pool.starmap

    total_dias = len(df_retornos)
    tamanho_treino = args.train_window

    print("============================================================")
    print(f"BACKTEST COM FILTRO DINAMICO (Top {int(args.quartile_filter * 100)}% Sharpe)")
    print(f"Ativos Brutos Iniciais: {len(universo_ativos_bruto)}")
    print(f"Ativos 100% Negociados e sem NAs: {len(ativos_negociados)}")
    print(f"Janela de Treino: {tamanho_treino} dias")
    print("Aplicacao: t+1")
    print("Rebalanceamento: diario")
    print("============================================================\n")

    # 3. Backtest rolling diario: estima em t e aplica em t+1.
    for idx_atual in range(tamanho_treino, total_dias):
        start_rebal = time.time()

        in_sample = df_retornos.iloc[idx_atual - tamanho_treino : idx_atual]
        linha_aplicacao = df_retornos.iloc[idx_atual]

        data_rebalanceamento = in_sample.index[-1]
        data_aplicacao = linha_aplicacao.name

        in_sample_assets = in_sample[universo_ativos]
        application_assets = linha_aplicacao[universo_ativos]
        in_sample_assets, application_assets, sanitizacao_stats = sanitize_returns(
            in_sample_assets,
            application_assets,
            args.outlier_method,
            args.winsor_limits,
            args.iqr_multiplier,
        )

        in_sample_f = in_sample_assets.copy()
        in_sample_f[rf_col] = in_sample[rf_col].values
        rf_dinamico = in_sample_f[rf_col].mean()
        ret_medios_full = in_sample_f[universo_ativos].mean().values
        cov_matrix_full = in_sample_f[universo_ativos].cov().values

        indices_restritos, sharpes_hist = build_dynamic_universe(
            ret_medios_full,
            cov_matrix_full,
            rf_dinamico,
            args.quartile_filter,
            args.k,
        )

        universo_restrito = [universo_ativos[i] for i in indices_restritos]
        sharpes_restritos = sharpes_hist[indices_restritos]
        ret_medios = ret_medios_full[indices_restritos]
        cov_matrix = cov_matrix_full[np.ix_(indices_restritos, indices_restritos)]

        ativos_validos = universo_restrito
        k_efetivo = min(args.k, len(ativos_validos))
        if k_efetivo == 0:
            raise ValueError("Nenhum ativo permaneceu no universo restrito dinamico.")

        pesos_novos_janela = np.zeros(len(ativos_validos))

        if args.strategy == "naive":
            top_k, p_naive = naive_1_k_allocation(ret_medios, cov_matrix, rf_dinamico, k_efetivo)
            pesos_novos_janela[top_k] = p_naive[top_k]
            optimization_status = "naive"
            fallback_reason = ""
        else:
            algorithm = BRKGA(
                n_elites=args.n_elites,
                n_offsprings=args.n_offsprings,
                n_mutants=args.n_mutants,
                bias=args.bias,
            )

            if args.strategy in ["rp_convex", "rp_nonconvex"]:
                form = "convex" if args.strategy == "rp_convex" else "non_convex"
                problem = RiskBudgetingBRKGA(
                    cov_matrix,
                    k_efetivo,
                    formulation=form,
                    solver_method=args.solver,
                    seed=args.seed,
                    **problem_kwargs,
                )
            elif args.strategy == "msr":
                problem = MaximumSharpeBRKGA(
                    ret_medios,
                    cov_matrix,
                    rf_dinamico,
                    k_efetivo,
                    **problem_kwargs,
                )
            elif args.strategy == "gmv":
                problem = MinimumVarianceBRKGA(cov_matrix, k_efetivo, **problem_kwargs)
            else:
                raise ValueError(f"Estrategia invalida: {args.strategy}")

            res = pymoo_minimize(
                problem,
                algorithm,
                ("n_gen", args.n_gen),
                seed=args.seed,
                verbose=args.verbose,
            )

            indices_sel = decode_brkga_solution_or_none(problem, getattr(res, "X", None), k_efetivo)
            brkga_success = indices_sel is not None
            fallback_reason = ""

            if not brkga_success:
                fallback_reason = "brkga_invalid_solution"
                indices_sel = fallback_subset_indices(args.strategy, ret_medios, cov_matrix, k_efetivo)

            sub_cov = cov_matrix[np.ix_(indices_sel, indices_sel)]

            if args.strategy == "rp_convex":
                res_c = minimize(
                    problem._obj_convex,
                    np.ones(k_efetivo) / k_efetivo,
                    args=(sub_cov, problem.b_target),
                    method=args.solver,
                    bounds=tuple((1e-8, None) for _ in range(k_efetivo)),
                )
                w_final = normalized_weights_or_none(getattr(res_c, "x", None))
            elif args.strategy == "rp_nonconvex":
                res_c = minimize(
                    problem._obj_non_convex,
                    np.ones(k_efetivo) / k_efetivo,
                    args=(sub_cov, problem.b_target),
                    method="SLSQP",
                    bounds=tuple((0.0, 1.0) for _ in range(k_efetivo)),
                    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                )
                w_final = normalized_weights_or_none(getattr(res_c, "x", None))
            elif args.strategy == "msr":
                res_c = minimize(
                    problem._neg_sharpe,
                    np.ones(k_efetivo) / k_efetivo,
                    args=(ret_medios[indices_sel], sub_cov, rf_dinamico),
                    method="SLSQP",
                    bounds=tuple((0.0, 1.0) for _ in range(k_efetivo)),
                    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                )
                w_final = normalized_weights_or_none(getattr(res_c, "x", None))
            else:
                res_c = minimize(
                    problem._obj_variance,
                    np.ones(k_efetivo) / k_efetivo,
                    args=(sub_cov,),
                    method="SLSQP",
                    bounds=tuple((0.0, 1.0) for _ in range(k_efetivo)),
                    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                )
                w_final = normalized_weights_or_none(getattr(res_c, "x", None))

            local_success = getattr(res_c, "success", False) and w_final is not None
            if not local_success:
                if fallback_reason:
                    fallback_reason += ";local_refinement_failed"
                else:
                    fallback_reason = "local_refinement_failed"
                w_final = np.ones(k_efetivo) / k_efetivo

            if brkga_success and local_success:
                optimization_status = "ok"
            elif brkga_success:
                optimization_status = "fallback_local"
            else:
                optimization_status = "fallback_brkga"

            pesos_novos_janela[indices_sel] = w_final

        tempo_janela = time.time() - start_rebal
        log_tempos_rebalanceamento.append(
            {
                "Data_Rebalanceamento": data_rebalanceamento,
                "Data_Aplicacao": data_aplicacao,
                "Tempo_Segundos": tempo_janela,
            }
        )
        log_universos_dinamicos.append(
            {
                "Data_Rebalanceamento": data_rebalanceamento,
                "Data_Aplicacao": data_aplicacao,
                "Tamanho_Universo_Restrito": len(universo_restrito),
                "Ativos_Universo_Restrito": ", ".join(
                    f"{ativo} ({sharpe:.4f})"
                    for ativo, sharpe in zip(universo_restrito, sharpes_restritos)
                ),
            }
        )
        log_sanitizacao.append(
            {
                "Data_Rebalanceamento": data_rebalanceamento,
                "Data_Aplicacao": data_aplicacao,
                "Metodo": args.outlier_method,
                "Obs_Clippadas_InSample": sanitizacao_stats["n_in_sample_clipped"],
                "Obs_Clippadas_Aplicacao": sanitizacao_stats["n_application_clipped"],
            }
        )
        log_otimizacao.append(
            {
                "Data_Rebalanceamento": data_rebalanceamento,
                "Data_Aplicacao": data_aplicacao,
                "Status_Otimizacao": optimization_status,
                "Motivo_Fallback": fallback_reason,
            }
        )

        pesos_novos_global = np.zeros(len(universo_ativos))
        for idx_local, idx_global in enumerate(indices_restritos):
            pesos_novos_global[idx_global] = pesos_novos_janela[idx_local]

        turnover = metrics.calculate_turnover(pesos_historicos, pesos_novos_global)
        custo_rebal = turnover * args.transaction_cost

        log_pesos_diarios.append(
            {"Data": data_aplicacao, **{a: pesos_novos_global[i] for i, a in enumerate(universo_ativos)}}
        )
        log_custos_diarios.append(
            {
                "Data_Rebalanceamento": data_rebalanceamento,
                "Data_Aplicacao": data_aplicacao,
                "Turnover": turnover,
                "Custo_Transacao": custo_rebal,
            }
        )

        retornos_aplicacao = application_assets.values
        ret_bruto = np.dot(pesos_novos_global, retornos_aplicacao)
        ret_liquido = ret_bruto - custo_rebal
        portfolio_oos_bruto.append(ret_bruto)
        portfolio_oos_liquido.append(ret_liquido)
        datas_oos_globais.append(data_aplicacao)

        w_drift = pesos_novos_global * (1 + retornos_aplicacao)
        soma_w = np.sum(w_drift)
        if soma_w > 1e-10:
            w_drift /= soma_w
        pesos_historicos = w_drift

        print(
            f"[*] [{args.strategy.upper()}] Rebal: {data_rebalanceamento.strftime('%Y-%m-%d')} "
            f"| Aplica: {data_aplicacao.strftime('%Y-%m-%d')} "
            f"| Universo Restrito: {len(ativos_validos)} ativos -> K={k_efetivo} "
            f"| Status: {optimization_status} "
            f"| Tempo: {tempo_janela:.2f}s"
        )

        if args.output_dir:
            pd.DataFrame(
                {
                    "Retorno_Bruto": portfolio_oos_bruto,
                    "Retorno_Liquido": portfolio_oos_liquido,
                    # Mantido por compatibilidade com scripts antigos.
                    "Retorno": portfolio_oos_liquido,
                },
                index=datas_oos_globais,
            ).to_csv(os.path.join(args.output_dir, f"oos_ts_{id_exp}.csv"))
            pd.DataFrame(log_tempos_rebalanceamento).to_csv(
                os.path.join(args.output_dir, f"exec_times_{id_exp}.csv"),
                index=False,
            )

    if pool:
        pool.close()
        pool.join()

    # 4. Calculo das metricas finais.
    total_exec = time.time() - start_total
    ts_bruto = pd.Series(portfolio_oos_bruto, index=datas_oos_globais)
    ts_liquido = pd.Series(portfolio_oos_liquido, index=datas_oos_globais)
    rf_oos = df_retornos.loc[datas_oos_globais, rf_col]

    if args.output_dir:
        df_res = pd.DataFrame(
            {
                "Estrategia": [args.strategy],
                "K": [args.k],
                "Quartil": [args.quartile_filter],
                "Retorno_Anual_Bruto": [metrics.annualized_return(ts_bruto)],
                "Vol_Anual_Bruto": [metrics.annualized_volatility(ts_bruto)],
                "Sharpe_Bruto": [metrics.sharpe_ratio(ts_bruto, rf_oos)],
                "Sortino_Bruto": [metrics.sortino_ratio(ts_bruto, rf_oos)],
                "MDD_Bruto": [metrics.maximum_drawdown(ts_bruto)],
                "Retorno_Anual_Liquido": [metrics.annualized_return(ts_liquido)],
                "Vol_Anual_Liquido": [metrics.annualized_volatility(ts_liquido)],
                "Sharpe_Liquido": [metrics.sharpe_ratio(ts_liquido, rf_oos)],
                "Sortino_Liquido": [metrics.sortino_ratio(ts_liquido, rf_oos)],
                "MDD_Liquido": [metrics.maximum_drawdown(ts_liquido)],
                "Tempo_Total_Seg": [total_exec],
            }
        )
        master_path = os.path.join(args.output_dir, "resultados_mestre.csv")
        df_res.to_csv(master_path, mode="a", header=not os.path.exists(master_path), index=False)

        pd.DataFrame(log_pesos_diarios).set_index("Data").to_csv(
            os.path.join(args.output_dir, f"pesos_diarios_{id_exp}.csv")
        )
        pd.DataFrame(log_custos_diarios).to_csv(
            os.path.join(args.output_dir, f"costs_daily_{id_exp}.csv"),
            index=False,
        )
        pd.DataFrame(log_universos_dinamicos).to_csv(
            os.path.join(args.output_dir, f"universos_dinamicos_{id_exp}.csv"),
            index=False,
        )
        pd.DataFrame(log_sanitizacao).to_csv(
            os.path.join(args.output_dir, f"sanitizacao_{id_exp}.csv"),
            index=False,
        )
        pd.DataFrame(log_otimizacao).to_csv(
            os.path.join(args.output_dir, f"otimizacao_{id_exp}.csv"),
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Solver - Otimizacao de Portfolios via BRKGA")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["rp_convex", "rp_nonconvex", "msr", "gmv", "naive"],
    )
    parser.add_argument("--solver", type=str, default="SLSQP")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--train_window", type=int, default=252)
    parser.add_argument("--test_window", type=int, default=1)
    parser.add_argument("--quartile_filter", type=float, default=1.0)
    parser.add_argument("--transaction_cost", type=float, default=0.005)
    parser.add_argument("--outlier_method", type=str, default="winsor", choices=["winsor", "iqr"])
    parser.add_argument("--winsor_limits", type=float, default=0.01)
    parser.add_argument("--iqr_multiplier", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--n_gen", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_elites", type=int, default=20)
    parser.add_argument("--n_offsprings", type=int, default=70)
    parser.add_argument("--n_mutants", type=int, default=10)
    parser.add_argument("--bias", type=float, default=0.7)
    parser.add_argument("--verbose", action="store_true")

    run_backtest(parser.parse_args())
