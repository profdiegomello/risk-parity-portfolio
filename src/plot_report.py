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

    if "Retorno_Bruto" in df.columns:
        serie_bruta = limpar_serie(df["Retorno_Bruto"].copy())
    else:
        serie_bruta = None

    if "Retorno_Liquido" in df.columns:
        serie_liquida = limpar_serie(df["Retorno_Liquido"].copy())
    elif "Retorno" in df.columns:
        serie_liquida = limpar_serie(df["Retorno"].copy())
    else:
        raise ValueError("arquivo sem coluna 'Retorno_Liquido' ou 'Retorno'")

    if serie_bruta is None:
        serie_bruta = serie_liquida.copy()

    return {"bruto": serie_bruta, "liquido": serie_liquida}


def write_validation_report(output_dir, filename, linhas):
    caminho = os.path.join(output_dir, filename)
    with open(caminho, "w", encoding="utf-8") as f:
        for linha in linhas:
            f.write(linha + "\n")
    print(f"[INFO] Relatorio de validacao salvo em: {caminho}")


def plot_wealth_index(output_dir, k_filter, quartile_filter, filename="wealth_index.png"):
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

    series_por_estrategia = {"bruto": {}, "liquido": {}}
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
            series = load_oos_series(arquivo)
            if label in series_por_estrategia["liquido"]:
                raise ValueError(f"estrategia duplicada no diretorio para o mesmo filtro: {label}")

            series_por_estrategia["bruto"][label] = series["bruto"]
            series_por_estrategia["liquido"][label] = series["liquido"]
            relatorio.append(
                f"[OK] {label}: {len(series['liquido'])} observacoes de {series['liquido'].index.min().date()} ate {series['liquido'].index.max().date()} ({nome_base})"
            )
        except Exception as e:
            mensagem = f"[AVISO] Arquivo ignorado: {nome_base} -> {e}"
            print(mensagem)
            relatorio.append(mensagem)

    if not series_por_estrategia["liquido"]:
        print("[ERRO] Nenhum dado valido para plotagem.")
        write_validation_report(output_dir, "plot_validation_report.txt", relatorio)
        return

    faltantes = [label for label in ordem_esperada if label not in series_por_estrategia["liquido"]]
    if faltantes:
        relatorio.append("")
        relatorio.append(f"[AVISO] Estrategias esperadas ausentes ou invalidadas: {', '.join(faltantes)}")

    datas_min = {
        label: serie.index.min() for label, serie in series_por_estrategia["liquido"].items()
    }
    datas_max = {
        label: serie.index.max() for label, serie in series_por_estrategia["liquido"].items()
    }
    tamanhos = {
        label: len(serie) for label, serie in series_por_estrategia["liquido"].items()
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

    dfs_acumulados = {"bruto": pd.DataFrame(), "liquido": pd.DataFrame()}
    for tipo in ["bruto", "liquido"]:
        for label in ordem_esperada:
            if label not in series_por_estrategia[tipo]:
                continue

            serie = series_por_estrategia[tipo][label]
            serie_alinhada = serie.loc[(serie.index >= inicio_comum) & (serie.index <= fim_comum)]
            if serie_alinhada.empty:
                relatorio.append(f"[AVISO] {label} ({tipo}) ficou vazia apos alinhamento e nao sera plotada.")
                continue

            retorno_acumulado = serie_alinhada.cumsum()
            dfs_acumulados[tipo][label] = retorno_acumulado

            if len(serie_alinhada) != tamanhos[label]:
                relatorio.append(
                    f"[AVISO] {label} ({tipo}) foi truncada para a janela comum: {len(serie_alinhada)} de {tamanhos[label]} observacoes."
                )

    if dfs_acumulados["bruto"].empty or dfs_acumulados["liquido"].empty:
        print("[ERRO] Nenhum dado valido permaneceu apos alinhamento.")
        write_validation_report(output_dir, "plot_validation_report.txt", relatorio)
        return

    for tipo in ["bruto", "liquido"]:
        if dfs_acumulados[tipo].isna().any().any():
            relatorio.append(
                f"[AVISO] Existem valores ausentes apos alinhamento ({tipo}); linhas incompletas serao removidas."
            )
            dfs_acumulados[tipo] = dfs_acumulados[tipo].dropna(how="any")

    if dfs_acumulados["bruto"].empty or dfs_acumulados["liquido"].empty:
        print("[ERRO] O alinhamento produziu uma matriz vazia apos remover valores ausentes.")
        write_validation_report(output_dir, "plot_validation_report.txt", relatorio)
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for ax, tipo, subtitulo in [
        (axes[0], "bruto", "Antes do Custo de Transacao"),
        (axes[1], "liquido", "Apos o Custo de Transacao"),
    ]:
        df_acumulado = dfs_acumulados[tipo]
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
        ax.tick_params(axis="x", rotation=90, labelsize="x-small")
        ax.set_title(subtitulo, fontsize=11, fontweight="bold")
        ax.set_ylabel("Retorno Acumulado")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(
        f"Evolucao do Retorno Acumulado - K={k_filter}, Top {int(quartile_filter * 100)}% Sharpe",
        fontweight="bold",
    )
    axes[-1].set_xlabel("Data de Referencia")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=True)
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))
    caminho_saida = os.path.join(output_dir, filename)
    fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)

    relatorio.append("")
    relatorio.append(
        f"Series finais no painel bruto: {', '.join([c for c in ordem_esperada if c in dfs_acumulados['bruto'].columns])}"
    )
    relatorio.append(
        f"Series finais no painel liquido: {', '.join([c for c in ordem_esperada if c in dfs_acumulados['liquido'].columns])}"
    )
    relatorio.append(f"Numero final de datas plotadas (bruto): {len(dfs_acumulados['bruto'])}")
    relatorio.append(f"Numero final de datas plotadas (liquido): {len(dfs_acumulados['liquido'])}")

    write_validation_report(output_dir, "plot_validation_report.txt", relatorio)
    print(f"[INFO] Grafico gerado com sucesso: {caminho_saida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de Graficos SBPO 2026")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--quartile", type=float, required=True)
    parser.add_argument("--filename", type=str, default="comparativo_portfolio.png")

    args = parser.parse_args()
    plot_wealth_index(args.output_dir, args.k, args.quartile, args.filename)
