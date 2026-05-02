import argparse
import os
import pandas as pd

# Presume-se que o arquivo metrics.py esteja no mesmo diretorio
import metrics 

def compute_metrics(args):
    """
    Computa metricas de performance para uma serie temporal de portifolio pre-existente
    e as anexa ao arquivo 'resultados_mestre.csv'.
    """
    id_exp = f"{args.strategy}_K{args.k}_Q{int(args.quartile * 100)}"
    ts_file = os.path.join(args.output_dir, f"oos_ts_{id_exp}.csv")
    
    if not os.path.exists(ts_file):
        print(f"[ERRO] Arquivo de serie temporal nao encontrado: {ts_file}")
        return
        
    print(f"[INFO] Carregando serie temporal (OOS): {ts_file}")
    df_ts = pd.read_csv(ts_file, index_col=0, parse_dates=True)
    
    print(f"[INFO] Carregando dataset original para indexacao da taxa livre de risco: {args.input}")
    df_input = pd.read_csv(args.input, index_col=0, parse_dates=True)
    
    rf_col = "RISKFREE" if "RISKFREE" in df_input.columns else "CDI"
    if rf_col not in df_input.columns:
        raise ValueError("O dataset original deve conter 'RISKFREE' ou 'CDI'.")
        
    # Assegura a intersecção exata das datas avaliadas fora da amostra (OOS)
    datas_oos = df_ts.index.intersection(df_input.index)
    if len(datas_oos) == 0:
        print("[ERRO] Nenhuma data em comum entre a serie OOS gerada e o dataset original.")
        return
        
    df_ts = df_ts.loc[datas_oos]
    rf_oos = df_input.loc[datas_oos, rf_col]
    
    ts_bruto = df_ts["Retorno_Bruto"]
    ts_liquido = df_ts["Retorno_Liquido"]
    
    # Executa os calculos usando as funcoes estabelecidas em metrics.py
    df_res = pd.DataFrame(
        {
            "Estrategia": [args.strategy],
            "K": [args.k],
            "Quartil": [args.quartile],
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
            "Tempo_Total_Seg": [None], # Avaliacao post-hoc, sem registro de processamento computacional
        }
    )
    
    master_path = os.path.join(args.output_dir, "resultados_mestre.csv")
    
    # Anexa (append) ou cria o relatorio final preservando logs anteriores
    df_res.to_csv(master_path, mode="a", header=not os.path.exists(master_path), index=False)
    
    print(f"[OK] Metricas calculadas com sucesso para {id_exp}.")
    print(f"[OK] Linha anexada no relatorio: {master_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compilador Isolado de Metricas de Portfolio")
    
    parser.add_argument("--input", type=str, required=True, help="Caminho para o CSV original (base de retornos e RISKFREE)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Diretório alvo contendo a subpasta de outputs da otimizacao")
    parser.add_argument("--strategy", type=str, required=True, choices=["rp_convex", "rp_nonconvex", "msr", "gmv", "naive"], help="Estrategia alvo analisada")
    parser.add_argument("--k", type=int, required=True, help="Restricao de cardinalidade K parametrizada")
    parser.add_argument("--quartile", type=float, required=True, help="Fator decimal do quartil (ex: 1.0, 0.5)")
    
    args = parser.parse_args()
    compute_metrics(args)
