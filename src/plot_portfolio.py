import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


def load_oos_series(arquivo):
    df = pd.read_csv(arquivo, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError("arquivo vazio")

    if df.shape[1] == 0:
        raise ValueError("arquivo sem coluna de retorno")

    def limpar_serie(serie):
        serie = pd.to_numeric(serie, errors="coerce")
        serie = serie.dropna()

        if serie.empty:
            raise ValueError("serie de retornos vazia apos limpeza")

        if serie.index.has_duplicates:
            serie = serie.groupby(level=0).last()

        serie = serie.sort_index()

        if not np.isfinite(serie.values).all():
            raise ValueError("serie contem valores nao finitos")

        return serie

    if "Retorno_Liquido" in df.columns:
        serie_liquida = limpar_serie(df["Retorno_Liquido"].copy())
    elif "Retorno" in df.columns:
        serie_liquida = limpar_serie(df["Retorno"].copy())
    else:
        raise ValueError("arquivo sem coluna 'Retorno_Liquido' ou 'Retorno'")

    return serie_liquida


def write_validation_report(output_dir, filename, linhas):
    caminho = os.path.join(output_dir, filename)
    with open(caminho, "w", encoding="utf-8") as f:
        for linha in linhas:
            f.write(linha + "\n")
    print(f"[INFO] Relatorio de validacao salvo em: {caminho}")


def plot_wealth_index(output_dir, k_filter, quartile_filter, filename="wealth_index.png", width=12, height=6, font_size="x-small"):
    padrao_busca = os.path.join(output_dir, f"oos_ts_*_K{k_filter}_Q{int(quartile_filter * 100)}.csv")
    arquivos = sorted(glob.glob(padrao_busca))

    if not arquivos:
        print(f"[ERRO] Nenhum arquivo encontrado para K={k_filter} e Quartil={quartile_filter} no diretorio {output_dir}")
        return

    mapeamento_nomes = {
        "rp_convex": "Risk Parity (Convex)",
        "rp_nonconvex": "Risk Parity (Non-Convex)",
        "msr": "Max Sharpe (Markowitz)",
        "gmv": "Min Variance (GMV)",
        "naive": "Naive (1/K)",
    }
    ordem_esperada = [
        "Risk Parity (Convex)",
        "Risk Parity (Non-Convex)",
        "Min Variance (GMV)",
        "Naive (1/K)",
    ]

    series_por_estrategia = {}
    relatorio = [
        "RELATORIO DE VALIDACAO DO PLOT",
        f"Diretorio: {output_dir}",
        f"K: {k_filter}",
        f"Quartil: {quartile_filter}",
        "",
    ]

    for arquivo in arquivos:
        nome_base = os.path.basename(arquivo)
        estrategia = nome_base.split("_K")[0].replace("oos_ts_", "")
        label = mapeamento_nomes.get(estrategia, estrategia)

        if estrategia == "msr":
            relatorio.append(f"[INFO] Arquivo ignorado por configuracao do grafico: {nome_base}")
            continue

        try:
            serie = load_oos_series(arquivo)
            if label in series_por_estrategia:
                raise ValueError(f"estrategia duplicada no diretorio para o mesmo filtro: {label}")

            series_por_estrategia[label] = serie
            relatorio.append(
                f"[OK] {label}: {len(serie)} observacoes de {serie.index.min().date()} ate {serie.index.max().date()} ({nome_base})"
            )
        except Exception as e:
            mensagem = f"[AVISO] Arquivo ignorado: {nome_base} -> {e}"
            print(mensagem)
            relatorio.append(mensagem)

    if not series_por_estrategia:
        print("[ERRO] Nenhum dado valido para plotagem.")
        write_validation_report(output_dir, "plot_validation_report.txt", relatorio)
        return

    faltantes = [label for label in ordem_esperada if label not in series_por_estrategia]
    if faltantes:
        relatorio.append("")
        relatorio.append(f"[AVISO] Estrategias esperadas ausentes ou invalidadas: {', '.join(faltantes)}")

    datas_min = {
        label: serie.index.min() for label, serie in series_por_estrategia.items()
    }
    datas_max = {
        label: serie.index.max() for label, serie in series_por_estrategia.items()
    }
    tamanhos = {
        label: len(serie) for label, serie in series_por_estrategia.items()
    }

    inicio_comum = max(datas_min.values())
    fim_comum = min(datas_max.values())

    if inicio_comum > fim_comum:
        relatorio.append("")
        relatorio.append("[ERRO] As series validas nao possuem intersecao de datas.")
        print("[ERRO] As series validas nao possuem intersecao de datas.")
        write_validation_report(output_dir, "plot_validation_report.txt", relatorio)
        return

    relatorio.append("")
    relatorio.append(f"Janela comum usada no grafico: {inicio_comum.date()} ate {fim_comum.date()}")

    df_acumulado = pd.DataFrame()
    for label in ordem_esperada:
        if label not in series_por_estrategia:
            continue

        serie = series_por_estrategia[label]
        serie_alinhada = serie.loc[(serie.index >= inicio_comum) & (serie.index <= fim_comum)]
        if serie_alinhada.empty:
            relatorio.append(f"[AVISO] {label} ficou vazia apos alinhamento e nao sera plotada.")
            continue

        retorno_acumulado = serie_alinhada.cumsum()
        df_acumulado[label] = retorno_acumulado

        if len(serie_alinhada) != tamanhos[label]:
            relatorio.append(
                f"[AVISO] {label} foi truncada para a janela comum: {len(serie_alinhada)} de {tamanhos[label]} observacoes."
            )

    if df_acumulado.empty:
        print("[ERRO] Nenhum dado valido permaneceu apos alinhamento.")
        write_validation_report(output_dir, "plot_validation_report.txt", relatorio)
        return

    if df_acumulado.isna().any().any():
        relatorio.append(
            f"[AVISO] Existem valores ausentes apos alinhamento; linhas incompletas serao removidas."
        )
        df_acumulado = df_acumulado.dropna(how="any")

    if df_acumulado.empty:
        print("[ERRO] O alinhamento produziu uma matriz vazia apos remover valores ausentes.")
        write_validation_report(output_dir, "plot_validation_report.txt", relatorio)
        return

    fig, ax = plt.subplots(figsize=(width, height))

    for coluna in [c for c in ordem_esperada if c in df_acumulado.columns]:
        if "Risk Parity" in coluna:
            ax.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, linewidth=2)
        elif "Naive" in coluna:
            ax.plot(
                df_acumulado.index,
                df_acumulado[coluna],
                label=coluna,
                linestyle="--",
                color="gray",
                alpha=0.7,
            )
        else:
            ax.plot(df_acumulado.index, df_acumulado[coluna], label=coluna, alpha=0.8)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
    ax.tick_params(axis="x", rotation=90, labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.set_ylabel("Retorno Acumulado", fontsize=font_size)
    ax.set_xlabel("Data de Referencia", fontsize=font_size)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.grid(True, alpha=0.3, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    # Aplicando font_size à legenda via parâmetro fontsize
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=4, frameon=True, fontsize=font_size)
    
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))
    caminho_saida = os.path.join(output_dir, filename)
    fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)

    relatorio.append("")
    relatorio.append(
        f"Series finais no painel: {', '.join([c for c in ordem_esperada if c in df_acumulado.columns])}"
    )
    relatorio.append(f"Numero final de datas plotadas: {len(df_acumulado)}")

    write_validation_report(output_dir, "plot_validation_report.txt", relatorio)
    print(f"[INFO] Grafico gerado com sucesso: {caminho_saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de Graficos SBPO 2026")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--quartile", type=float, required=True)
    parser.add_argument("--filename", type=str, default="plot_portfolio.png")
    parser.add_argument("--width", type=float, default=12.0)
    parser.add_argument("--height", type=float, default=6.0)
    parser.add_argument("--font", type=str, default="x-small")

    args = parser.parse_args()
    plot_wealth_index(
        args.output_dir, 
        args.k, 
        args.quartile, 
        args.filename, 
        args.width, 
        args.height, 
        args.font
    )
