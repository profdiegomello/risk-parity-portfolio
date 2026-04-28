import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import matplotlib.ticker as ticker

def run_analysis(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Carregando dados de: {args.input}")
    df = pd.read_csv(args.input)

    df['Convergence_Gap'] = np.abs(df['rp_convex'] - df['rp_nonconvex'])
    df['Total_Pop'] = df['n_elites'] + df['n_offsprings'] + df['n_mutants']

    group_cols = ['n_elites', 'n_offsprings', 'n_mutants', 'n_gen', 'Total_Pop']
    
    summary = df.groupby(group_cols).agg(
        Mean_Gap=('Convergence_Gap', 'mean'),
        Std_Gap=('Convergence_Gap', 'std'),
        Max_Gap=('Convergence_Gap', 'max'),
        Mean_Time=('Time_Sec', 'mean'),
        N_Rodadas=('Data', 'count')
    ).reset_index()

    summary = summary.sort_values(by=['Mean_Gap', 'Mean_Time']).reset_index(drop=True)

    caminho_csv = os.path.join(args.output_dir, "param_ranking_analysis.csv")
    summary.to_csv(caminho_csv, index=False)
    
    print("\n" + "="*80)
    print("TOP 10 COMBINAÇÕES: CONVERGÊNCIA DA PARIDADE DE RISCO")
    print("="*80)
    print(summary.head(10).to_string(index=False))
    print("="*80)

    # =========================================================================
    # MAPEAMENTO DE RÓTULOS E LIMITES DE ESCALA
    # =========================================================================
    label_map = {
        'n_gen': 'Gerações',
        'n_elites': 'Elites',
        'n_offsprings': 'Offsprings',
        'n_mutants': 'Mutantes',
        'Total_Pop': 'População Total',
        'Mean_Time': 'Tempo Médio (s)',
        'Mean_Gap': 'Gap Médio',
        'Convergence_Gap': 'Gap Absoluto',
        'Time_Sec': 'Tempo (s)'
    }
    
    df_plot = df.rename(columns=label_map)
    summary_plot = summary.rename(columns=label_map)

    # Limites para a escala Log (geral)
    y_min_log = summary_plot['Gap Médio'].min() * 0.5
    y_max_log = summary_plot['Gap Médio'].max() * 1.5

    # Limites para a escala Linear (foco nas convergências válidas)
    valores_validos = summary_plot[summary_plot['Gap Médio'] < 0.1]['Gap Médio']
    y_max_linear = valores_validos.quantile(0.95) * 1.2 if not valores_validos.empty else summary_plot['Gap Médio'].max()

    def formatar_eixo_log(ax):
        ax.set_yscale('log')
        ax.set_ylim(bottom=y_min_log, top=y_max_log)
        locmaj = ticker.LogLocator(base=10.0, numticks=10) 
        ax.yaxis.set_major_locator(locmaj)
        locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1, numticks=10)
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.5)

    def formatar_eixo_linear(ax):
        ax.set_yscale('linear')
        ax.set_ylim(0, y_max_linear)
        ax.grid(True, ls="--", linewidth=0.5, alpha=0.5)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # =========================================================================
    # 1. Fronteira de Pareto Clássica (Tempo vs Gap)
    # =========================================================================
    # 1A. Escala Log
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=summary_plot, x='Tempo Médio (s)', y='Gap Médio', 
        hue='Gerações', size='Offsprings', sizes=(40, 250), 
        palette='viridis', alpha=0.7, edgecolor='black', linewidth=0.5, ax=ax
    )
    formatar_eixo_log(ax)
    plt.title('Trade-off de Convergência: Gap vs Tempo [Escala Log]', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plot_01a_pareto_classic_log.png"), dpi=300)
    plt.close()

    # 1B. Escala Linear
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=summary_plot, x='Tempo Médio (s)', y='Gap Médio', 
        hue='Gerações', size='Offsprings', sizes=(40, 250), 
        palette='viridis', alpha=0.7, edgecolor='black', linewidth=0.5, ax=ax
    )
    formatar_eixo_linear(ax)
    plt.title('Trade-off de Convergência: Gap vs Tempo [Escala Linear]', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plot_01b_pareto_classic_linear.png"), dpi=300)
    plt.close()

    # =========================================================================
    # 2. Painel Multi-Dimensional (FacetGrid)
    # =========================================================================
    # 2A. Escala Log
    g_log = sns.relplot(
        data=summary_plot, x='Tempo Médio (s)', y='Gap Médio',
        hue='Gerações', size='Offsprings', sizes=(20, 150), 
        palette='viridis', alpha=0.7, edgecolor='black', linewidth=0.3,
        col='Elites', row='Mutantes', 
        height=3.5, aspect=1.2, facet_kws={'margin_titles': True}
    )
    for ax in g_log.axes.flat: formatar_eixo_log(ax)
    g_log.fig.suptitle('Impacto das Subpopulações e Gerações [Escala Log]', fontweight='bold', y=1.02)
    plt.savefig(os.path.join(args.output_dir, "plot_02a_pareto_multidimensional_log.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2B. Escala Linear
    g_lin = sns.relplot(
        data=summary_plot, x='Tempo Médio (s)', y='Gap Médio',
        hue='Gerações', size='Offsprings', sizes=(20, 150), 
        palette='viridis', alpha=0.7, edgecolor='black', linewidth=0.3,
        col='Elites', row='Mutantes', 
        height=3.5, aspect=1.2, facet_kws={'margin_titles': True}
    )
    for ax in g_lin.axes.flat: formatar_eixo_linear(ax)
    g_lin.fig.suptitle('Impacto das Subpopulações e Gerações [Escala Linear]', fontweight='bold', y=1.02)
    plt.savefig(os.path.join(args.output_dir, "plot_02b_pareto_multidimensional_linear.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 3. Pareto Agregado (Tempo vs Gap por População Total)
    # =========================================================================
    # 3A. Escala Log
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=summary_plot, x='Tempo Médio (s)', y='Gap Médio', 
        hue='Gerações', size='População Total', sizes=(40, 300), 
        palette='magma', alpha=0.7, edgecolor='black', linewidth=0.5, ax=ax
    )
    formatar_eixo_log(ax)
    plt.title('Dinâmica Populacional: Gap vs Tempo [Escala Log]', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plot_03a_pareto_population_log.png"), dpi=300)
    plt.close()

    # 3B. Escala Linear
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=summary_plot, x='Tempo Médio (s)', y='Gap Médio', 
        hue='Gerações', size='População Total', sizes=(40, 300), 
        palette='magma', alpha=0.7, edgecolor='black', linewidth=0.5, ax=ax
    )
    formatar_eixo_linear(ax)
    plt.title('Dinâmica Populacional: Gap vs Tempo [Escala Linear]', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plot_03b_pareto_population_linear.png"), dpi=300)
    plt.close()

    # =========================================================================
    # 4. Matriz de Correlação
    # =========================================================================
    plt.figure(figsize=(8, 6))
    cols_to_correlate = ['Elites', 'Offsprings', 'Mutantes', 'População Total', 'Gerações', 'Gap Absoluto', 'Tempo (s)']
    corr_matrix = df_plot[cols_to_correlate].corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
    plt.title('Correlação de Spearman: Parâmetros vs Desempenho', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "plot_04_correlation_matrix.png"), dpi=300)
    plt.close()

    print(f"7 Gráficos gerados com sucesso na pasta: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Análise de Convergência de Hiperparâmetros')
    parser.add_argument('--input', type=str, required=True, help="Caminho para o param_comparison_master.csv")
    parser.add_argument('--output_dir', type=str, default='./param_analysis', help="Pasta para salvar os gráficos")
    
    args = parser.parse_args()
    run_analysis(args)
