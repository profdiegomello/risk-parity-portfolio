import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Leitura e alinhamento das séries temporais
    for arquivo in arquivos:
        # Extrai a estratégia do nome do arquivo
        nome_base = os.path.basename(arquivo)
        estrategia = nome_base.split('_K')[0].replace('oos_ts_', '')
        label = mapeamento_nomes.get(estrategia, estrategia)
        
        df = pd.read_csv(arquivo, index_col=0, parse_dates=True)
        # Calcula o índice de riqueza (Capital Inicial = R$ 1,00)
        df_acumulado[label] = (1 + df['Retorno']).cumprod()

    # Configuração estética acadêmica
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    # Plotagem com filtros de nomes estritos para evitar sobreposição de estilos
    for coluna in df_acumulado.columns:
        if 'Markowitz' in coluna:
            plt.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, linestyle=':', color='crimson', linewidth=2)
        elif 'GMV' in coluna:
            plt.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, linestyle='--', color='darkmagenta', linewidth=2)
        elif 'Naive' in coluna:
            plt.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, linestyle='-.', color='gray', linewidth=2)
        elif 'Non-Convex' in coluna:
            plt.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, linestyle='-', color='darkorange', linewidth=2)
        elif 'Convex' in coluna:
            plt.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, linestyle='-', color='navy', linewidth=2.5)
        else:
            plt.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, linestyle='-', color='black', linewidth=1)

    plt.title(f"Evolução do Capital Acumulado (Wealth Index) - K={k_filter}, Top {int(quartile_filter*100)}% Sharpe", fontweight='bold')
    plt.ylabel("Capital Indexado (Base = 1.0)")
    plt.xlabel("Tempo (Data de Rebalanceamento)")
    plt.axhline(1.0, color='black', linewidth=1)
    
    # Legenda alocada no rodapé (fora do gráfico), em duas colunas
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    
    plt.tight_layout()
    caminho_saida = os.path.join(output_dir, filename)
    plt.savefig(caminho_saida, dpi=300, bbox_inches='tight')
    print(f"[INFO] Gráfico comparativo gerado: {caminho_saida}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gerador de Gráficos para o Artigo do SBPO')
    parser.add_argument('--output_dir', type=str, required=True, help='Diretório onde estão os CSVs gerados.')
    parser.add_argument('--k', type=int, required=True, help='Cardinalidade K usada no experimento para filtrar os arquivos.')
    parser.add_argument('--quartile', type=float, required=True, help='Filtro de quartil usado no experimento.')
    
    args = parser.parse_args()
    
    plot_wealth_index(args.output_dir, args.k, args.quartile, filename=f"comparativo_K{args.k}_Q{int(args.quartile*100)}.png")
