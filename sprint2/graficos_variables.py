"""
=============================================================================
Gráficos de análisis de variables seleccionadas — Sprint 2
Proyecto : Predicción de Satisfacción del Cliente — Olist
=============================================================================
Genera 4 gráficos sobre las 29 features seleccionadas:
  01 — Importancia por Random Forest (train)
  02 — Mapa de correlación
  03 — Distribución de top features por clase (insatisfecho vs satisfecho)
  04 — Lista resumen con estadísticas descriptivas

Entrada : data/master/X_train.csv, y_train.csv
          data/reportes/features_seleccionadas.json
Salida  : reports/var_01_importancia.png
          reports/var_02_correlacion.png
          reports/var_03_distribuciones.png
          reports/var_04_resumen_estadisticas.png
=============================================================================
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent
DATA     = ROOT / "data" / "master"
REPORTES = ROOT / "data" / "reportes"
REPORTS  = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42


def _save(fig, nombre: str):
    path = REPORTS / nombre
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> reports/{nombre}")


def cargar_datos():
    X_train = pd.read_csv(DATA / "X_train.csv")
    y_train = pd.read_csv(DATA / "y_train.csv").squeeze()
    with open(REPORTES / "features_seleccionadas.json", encoding="utf-8") as f:
        info = json.load(f)
    features = info["features"]
    X_train = X_train[features]
    print(f"  Datos cargados: {X_train.shape[0]:,} filas × {len(features)} features")
    print(f"  Insatisfechos (cls 0): {(y_train == 0).mean():.1%}  |  "
          f"Satisfechos (cls 1): {(y_train == 1).mean():.1%}")
    return X_train, y_train, features


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Importancia por Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def plot_importancia(X_train, y_train, features):
    print("\n[1] Calculando importancia RF...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        class_weight="balanced", random_state=SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    imp = rf.feature_importances_

    df_imp = pd.DataFrame({"Feature": features, "Importancia": imp})
    df_imp = df_imp.sort_values("Importancia", ascending=True)

    norm = mcolors.Normalize(vmin=df_imp["Importancia"].min(),
                             vmax=df_imp["Importancia"].max())
    cmap = plt.cm.Blues
    colores = [cmap(norm(v) * 0.7 + 0.3) for v in df_imp["Importancia"]]

    fig, ax = plt.subplots(figsize=(9, 10))
    bars = ax.barh(df_imp["Feature"], df_imp["Importancia"],
                   color=colores, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, df_imp["Importancia"]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=7.5)

    ax.set_xlabel("Importancia (MDI)", fontsize=10)
    ax.set_title(
        f"Importancia de las {len(features)} variables seleccionadas\n"
        "(Random Forest — entrenado en X_train)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlim(0, df_imp["Importancia"].max() * 1.18)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.tick_params(axis="y", labelsize=8.5)
    plt.tight_layout()
    _save(fig, "var_01_importancia.png")
    return df_imp.sort_values("Importancia", ascending=False)["Feature"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 02 — Mapa de correlación
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlacion(X_train, features):
    print("[2] Generando mapa de correlación...")
    corr = X_train[features].corr()

    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Correlación de Pearson")

    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(features, fontsize=7)

    # Anotar celdas con correlación alta (|r| >= 0.6)
    for i in range(len(features)):
        for j in range(len(features)):
            val = corr.values[i, j]
            if i != j and abs(val) >= 0.6:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=5.5, color="white" if abs(val) > 0.8 else "black")

    ax.set_title(
        f"Mapa de correlación — {len(features)} variables seleccionadas",
        fontsize=11, fontweight="bold", pad=12
    )
    plt.tight_layout()
    _save(fig, "var_02_correlacion.png")


# ─────────────────────────────────────────────────────────────────────────────
# 03 — Distribuciones de top variables por clase
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribuciones(X_train, y_train, features_ordenadas):
    print("[3] Generando distribuciones por clase (top 12 features)...")
    top12 = features_ordenadas[:12]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()

    mask_0 = y_train == 0   # insatisfecho
    mask_1 = y_train == 1   # satisfecho

    for i, feat in enumerate(top12):
        ax = axes[i]
        vals_0 = X_train.loc[mask_0, feat].dropna()
        vals_1 = X_train.loc[mask_1, feat].dropna()

        q1 = min(vals_0.quantile(0.01), vals_1.quantile(0.01))
        q99 = max(vals_0.quantile(0.99), vals_1.quantile(0.99))
        bins = np.linspace(q1, q99, 40)

        ax.hist(vals_0, bins=bins, alpha=0.55, color="#e74c3c",
                label=f"Insatisfecho (n={len(vals_0):,})", density=True)
        ax.hist(vals_1, bins=bins, alpha=0.55, color="#2980b9",
                label=f"Satisfecho (n={len(vals_1):,})", density=True)

        ax.set_title(feat, fontsize=8.5, fontweight="bold")
        ax.set_ylabel("Densidad", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=6)

    plt.suptitle(
        "Distribución de las 12 variables más importantes por clase\n"
        "(rojo = Insatisfecho, azul = Satisfecho)",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save(fig, "var_03_distribuciones.png")


# ─────────────────────────────────────────────────────────────────────────────
# 04 — Resumen estadístico visual (tabla con medias por clase)
# ─────────────────────────────────────────────────────────────────────────────

def plot_resumen_estadisticas(X_train, y_train, features_ordenadas):
    print("[4] Generando tabla resumen estadístico...")

    mask_0 = y_train == 0
    mask_1 = y_train == 1

    rows = []
    for feat in features_ordenadas:
        col = X_train[feat]
        m0 = X_train.loc[mask_0, feat].mean()
        m1 = X_train.loc[mask_1, feat].mean()
        diff_pct = ((m0 - m1) / (abs(m1) + 1e-9)) * 100
        rows.append({
            "Variable": feat,
            "Media cls 0\n(Insatisfecho)": round(m0, 3),
            "Media cls 1\n(Satisfecho)": round(m1, 3),
            "Δ%\n(0 vs 1)": round(diff_pct, 1),
            "Min": round(col.min(), 3),
            "Max": round(col.max(), 3),
            "Std": round(col.std(), 3),
        })

    df_stats = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.axis("off")

    col_labels = list(df_stats.columns)
    cell_data  = df_stats.values.tolist()

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    # Header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Filas alternadas
    for i in range(1, len(rows) + 1):
        color = "#ecf0f1" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

        # Colorear la columna Δ% según signo
        delta = rows[i - 1]["Δ%\n(0 vs 1)"]
        if abs(delta) > 10:
            table[i, 3].set_facecolor("#fadbd8" if delta > 0 else "#d5f5e3")

    ax.set_title(
        "Resumen estadístico de variables — comparación por clase\n"
        "(ordenadas por importancia RF, de mayor a menor)",
        fontsize=11, fontweight="bold", pad=20
    )
    plt.tight_layout()
    _save(fig, "var_04_resumen_estadisticas.png")


# ─────────────────────────────────────────────────────────────────────────────
# Lista impresa en consola
# ─────────────────────────────────────────────────────────────────────────────

def imprimir_lista(features):
    print("\n" + "=" * 55)
    print(f"VARIABLES SELECCIONADAS ({len(features)} features)")
    print("=" * 55)
    for i, f in enumerate(features, 1):
        print(f"  {i:>2}. {f}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("graficos_variables.py — Análisis de variables Sprint 2")
    print("=" * 55)

    X_train, y_train, features = cargar_datos()
    imprimir_lista(features)

    features_ordenadas = plot_importancia(X_train, y_train, features)
    plot_correlacion(X_train, features)
    plot_distribuciones(X_train, y_train, features_ordenadas)
    plot_resumen_estadisticas(X_train, y_train, features_ordenadas)

    print("\nOK Gráficos generados en reports/")
    print("   var_01_importancia.png      — ranking de importancia RF")
    print("   var_02_correlacion.png      — mapa de correlaciones")
    print("   var_03_distribuciones.png   — distribución top 12 por clase")
    print("   var_04_resumen_estadisticas.png — tabla estadística")


if __name__ == "__main__":
    main()
