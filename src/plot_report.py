import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import glob
import argparse

def plot_wealth_index(output_dir, k_filter, quartile_filter, filename="wealth_index.png"):
    # Busca os arquivos que correspondem aos parâmetros do experimento
    padrao_busca = os.path.join(output_dir, f"oos_ts_*_K{k_filter}_Q{int(quartile_filter*100)}.csv")
    arquivos = glob.glob(padrao_busca)
    
    if not arquivos:
        print(f"[ERRO] Nenhum arquivo encontrado para K={k_filter} e Quartil={quartile_filter} no diretório {output_dir}")
        return

    df_acumulado = pd.DataFrame()

    mapeamento_nomes = {
        'rp_convex': 'Risk Parity (Convex)',
        'rp_nonconvex': 'Risk Parity (Non-Convex)',
        'msr': 'Max Sharpe (Markowitz)',
        'gmv': 'Min Variance (GMV)',
        'naive': 'Naive (1/K)'
    }

    for arquivo in arquivos:
        try:
            nome_base = os.path.basename(arquivo)
            estrategia = nome_base.split('_K')[0].replace('oos_ts_', '')
            label = mapeamento_nomes.get(estrategia, estrategia)
            
            df = pd.read_csv(arquivo, index_col=0, parse_dates=True)
            
            if df.empty:
                continue
                
            # Cálculo do Wealth Index (Capital Acumulado)
            serie_acumulada = (1 + df).cumprod()
            df_acumulado[label] = serie_acumulada.iloc[:, 0]
        except Exception as e:
            print(f"[AVISO] Erro ao processar {arquivo}: {e}. Pulando modelo.")
            continue

    if df_acumulado.empty:
        print("[ERRO] Nenhum dado válido para plotagem.")
        return

    plt.figure(figsize=(12, 7))
    
    for coluna in df_acumulado.columns:
        if 'Risk Parity' in coluna:
            plt.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, linewidth=2)
        elif 'Naive' in coluna:
            plt.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, linestyle='--', color='gray', alpha=0.7)
        else:
            plt.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, alpha=0.8)

    # Configuração do Eixo X (Datas)
    ax = plt.gca()
    # Rotaciona 90 graus, diminui a fonte e alinha verticalmente
    plt.xticks(rotation=90, fontsize='x-small')
    
    # Garante que as datas apareçam de forma legível independente da janela
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20)) 
    
    plt.title(f"Evolução do Capital Acumulado (Wealth Index) - K={k_filter}, Top {int(quartile_filter*100)}% Sharpe", fontweight='bold')
    plt.ylabel("Capital Indexado (Base = 1.0)")
    plt.xlabel("Data de Referência")
    plt.axhline(1.0, color='black', linewidth=0.8, linestyle=':')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    caminho_saida = os.path.join(output_dir, filename)
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    print(f"[INFO] Gráfico gerado com sucesso: {caminho_saida}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gerador de Gráficos SBPO 2026')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--quartile', type=float, required=True)
    parser.add_argument('--filename', type=str, default="comparativo_portfolio.png")
    
    args = parser.parse_args()
    plot_wealth_index(args.output_dir, args.k, args.quartile, args.filename)
