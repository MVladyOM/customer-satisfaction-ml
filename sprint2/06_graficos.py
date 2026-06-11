"""
=============================================================================
Script 06 - Gráficos de Performance y Feature Importance
Proyecto : Predicción de Satisfacción del Cliente — Olist
Grupo 1  | Sprint 2
=============================================================================
Responsabilidad:
  Genera todos los gráficos de análisis post-modelado:

  A) Performance comparativa
     1.  Barras comparativas de métricas (val + backtest)
     2.  Curvas ROC — todos los modelos en val
     3.  Curvas ROC — todos los modelos en backtest
     4.  Curvas Precision-Recall — val
     5.  Curvas Precision-Recall — backtest
     6.  Matrices de confusión (5 modelos × val)
     7.  Matrices de confusión (5 modelos × backtest)
     8.  Distribución de probabilidades predichas — modelo ganador

  B) Análisis Optuna
     9.  Convergencia de trials por modelo
     10. Importancia de hiperparámetros (Optuna FAnova)

  C) Feature importance
     11. Importancia nativa de features — modelos basados en árboles
     12. Top-15 features comparativo entre modelos de árbol

Entrada : data/master/X_val.csv / y_val.csv
          data/master/X_backtest.csv / y_backtest.csv
          data/reportes/resultados_optuna.json
          data/reportes/modelos_optuna.pkl
          data/reportes/estudios_optuna.pkl

Salida  : reports/01_comparacion_metricas.png
          reports/02_roc_val.png
          reports/03_roc_backtest.png
          reports/04_pr_val.png
          reports/05_pr_backtest.png
          reports/06_confusion_val.png
          reports/07_confusion_backtest.png
          reports/08_dist_probabilidades.png
          reports/09_optuna_convergencia.png
          reports/10_optuna_hiperparametros.png
          reports/11_feature_importance_nativa.png
          reports/12_feature_importance_comparativo.png
          data/reportes/estado_graficos.json
=============================================================================
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
MASTER_PATH   = os.path.join("data", "master")
REPORTES_PATH = os.path.join("data", "reportes")
GRAFICOS_PATH = "reports"

# ---------------------------------------------------------------------------
# Paleta de colores — uno por modelo
# ---------------------------------------------------------------------------
COLORES = {
    "Logistic Regression": "#4C72B0",
    "Random Forest"      : "#55A868",
    "Gradient Boosting"  : "#C44E52",
    "XGBoost"            : "#DD8452",
    "LightGBM"           : "#8172B2",
}
MODELOS_ARBOL = ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------

def cargar_todo() -> tuple:
    # Splits
    X_val      = pd.read_csv(os.path.join(MASTER_PATH, "X_val.csv"))
    y_val      = pd.read_csv(os.path.join(MASTER_PATH, "y_val.csv")).squeeze()
    X_backtest = pd.read_csv(os.path.join(MASTER_PATH, "X_backtest.csv"))
    y_backtest = pd.read_csv(os.path.join(MASTER_PATH, "y_backtest.csv")).squeeze()

    # Resultados Optuna
    with open(os.path.join(REPORTES_PATH, "resultados_optuna.json"), encoding="utf-8") as f:
        resultados = json.load(f)

    # Modelos entrenados + predicciones
    with open(os.path.join(REPORTES_PATH, "modelos_optuna.pkl"), "rb") as f:
        modelos = pickle.load(f)

    # Estudios Optuna
    with open(os.path.join(REPORTES_PATH, "estudios_optuna.pkl"), "rb") as f:
        estudios = pickle.load(f)

    nombres   = [r["modelo"] for r in resultados]
    ganador   = max(resultados, key=lambda r: r["f1_0_val"])["modelo"]
    features  = X_val.columns.tolist()

    print(f"  Modelos cargados  : {nombres}")
    print(f"  Modelo ganador    : {ganador}")
    print(f"  Features          : {len(features)}")
    print(f"  X_val shape       : {X_val.shape}")
    print(f"  X_backtest shape  : {X_backtest.shape}")

    return (X_val, y_val, X_backtest, y_backtest,
            resultados, modelos, estudios, nombres, ganador, features)


def guardar(fig, nombre_archivo: str, dpi: int = 150):
    ruta = os.path.join(GRAFICOS_PATH, nombre_archivo)
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {nombre_archivo}")


# ---------------------------------------------------------------------------
# A1. BARRAS COMPARATIVAS DE MÉTRICAS
# ---------------------------------------------------------------------------

def grafico_comparacion_metricas(resultados: list):
    metricas_val = ["f1_0_val", "recall_0_val", "prec_0_val", "f1_macro_val", "auc_val"]
    metricas_bt  = ["f1_0_bt",  "recall_0_bt",  "prec_0_bt",  "f1_macro_bt",  "auc_bt"]
    etiquetas    = ["F1 cls 0", "Recall cls 0", "Prec cls 0", "F1 macro", "AUC-ROC"]

    nombres = [r["modelo"] for r in resultados]
    colores = [COLORES[n] for n in nombres]
    x       = np.arange(len(etiquetas))
    ancho   = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (ax, cols, titulo) in enumerate(
        zip(axes, [metricas_val, metricas_bt], ["Validación", "Backtest"])
    ):
        for i, (r, color) in enumerate(zip(resultados, colores)):
            vals = [r[c] for c in cols]
            bars = ax.bar(x + i * ancho, vals, ancho, label=r["modelo"],
                          color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

        ax.set_title(f"Métricas — {titulo}", fontsize=12, fontweight="bold")
        ax.set_xticks(x + ancho * (len(nombres) - 1) / 2)
        ax.set_xticklabels(etiquetas, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Comparación de Modelos post-Optuna", fontsize=14, fontweight="bold")
    plt.tight_layout()
    guardar(fig, "01_comparacion_metricas.png")


# ---------------------------------------------------------------------------
# A2-A3. CURVAS ROC
# ---------------------------------------------------------------------------

def grafico_roc(modelos: dict, y_true, nombre_set: str, archivo: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    for nombre, datos in modelos.items():
        proba = np.array(datos[f"proba_{'val' if 'val' in archivo else 'bt'}"])
        fpr, tpr, _ = roc_curve(y_true, 1 - proba)  # 1-proba porque pos_label=0
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc:.3f})",
                color=COLORES[nombre], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=11)
    ax.set_title(f"Curvas ROC — {nombre_set} (clase: Insatisfecho)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    guardar(fig, archivo)


# ---------------------------------------------------------------------------
# A4-A5. CURVAS PRECISION-RECALL
# ---------------------------------------------------------------------------

def grafico_pr(modelos: dict, y_true, nombre_set: str, archivo: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    sufijo = "val" if "val" in archivo else "bt"

    for nombre, datos in modelos.items():
        proba = np.array(datos[f"proba_{sufijo}"])
        # invertimos proba para clase 0
        prec, rec, _ = precision_recall_curve(y_true, 1 - proba, pos_label=0)
        ap = average_precision_score(y_true, 1 - proba, pos_label=0)
        ax.plot(rec, prec, label=f"{nombre} (AP={ap:.3f})",
                color=COLORES[nombre], linewidth=2)

    baseline = (y_true == 0).mean()
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall (clase Insatisfecho)", fontsize=11)
    ax.set_ylabel("Precision (clase Insatisfecho)", fontsize=11)
    ax.set_title(f"Curvas Precision-Recall — {nombre_set}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    guardar(fig, archivo)


# ---------------------------------------------------------------------------
# A6-A7. MATRICES DE CONFUSIÓN
# ---------------------------------------------------------------------------

def grafico_confusion(modelos: dict, y_true, nombre_set: str, archivo: str):
    sufijo = "val" if "val" in archivo else "bt"
    n      = len(modelos)
    ncols  = 3
    nrows  = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    axes = axes.flatten()

    for i, (nombre, datos) in enumerate(modelos.items()):
        pred = np.array(datos[f"pred_{sufijo}"])
        cm   = confusion_matrix(y_true, pred)
        disp = ConfusionMatrixDisplay(
            cm, display_labels=["Insatisfecho", "Satisfecho"]
        )
        disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
        axes[i].set_title(nombre, fontsize=10, fontweight="bold")
        axes[i].tick_params(axis="x", labelrotation=15)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Matrices de Confusión — {nombre_set}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    guardar(fig, archivo)


# ---------------------------------------------------------------------------
# A8. DISTRIBUCIÓN DE PROBABILIDADES — MODELO GANADOR
# ---------------------------------------------------------------------------

def grafico_dist_probabilidades(modelos: dict, ganador: str,
                                  y_val, y_backtest):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, sufijo, y_true, titulo in zip(
        axes,
        ["val", "bt"],
        [y_val, y_backtest],
        ["Validación", "Backtest"],
    ):
        proba = np.array(modelos[ganador][f"proba_{sufijo}"])
        # proba es P(clase=1=satisfecho). Para insatisfecho usamos 1-proba
        proba_insatisfecho = 1 - proba

        mask_0 = y_true == 0
        mask_1 = y_true == 1

        ax.hist(proba_insatisfecho[mask_0], bins=40, alpha=0.6,
                color="#C44E52", label="Insatisfecho (real)", density=True)
        ax.hist(proba_insatisfecho[mask_1], bins=40, alpha=0.6,
                color="#4C72B0", label="Satisfecho (real)", density=True)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Umbral 0.5")
        ax.set_xlabel("P(Insatisfecho)", fontsize=11)
        ax.set_ylabel("Densidad", fontsize=11)
        ax.set_title(f"Distribución de Probabilidades — {titulo}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Modelo Ganador: {ganador}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    guardar(fig, "08_dist_probabilidades.png")


# ---------------------------------------------------------------------------
# B9. CONVERGENCIA DE OPTUNA
# ---------------------------------------------------------------------------

def grafico_convergencia(estudios: dict):
    n     = len(estudios)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, (nombre, study) in enumerate(estudios.items()):
        valores          = [t.value for t in study.trials if t.value is not None]
        mejor_acumulado  = np.maximum.accumulate(valores)
        ax               = axes[i]

        ax.plot(valores, alpha=0.35, color=COLORES.get(nombre, "steelblue"),
                linewidth=0.8, label="Trial")
        ax.plot(mejor_acumulado, color="crimson", linewidth=2,
                label="Mejor acumulado")
        ax.axhline(mejor_acumulado[-1], color="gray", linestyle=":",
                   linewidth=1, alpha=0.7)
        ax.set_title(nombre, fontsize=10, fontweight="bold")
        ax.set_xlabel("Trial", fontsize=9)
        ax.set_ylabel("F1 clase 0", fontsize=9)
        ax.annotate(f"Max: {mejor_acumulado[-1]:.4f}",
                    xy=(len(mejor_acumulado) - 1, mejor_acumulado[-1]),
                    xytext=(-60, -15), textcoords="offset points",
                    fontsize=8, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Convergencia de Optuna por Modelo", fontsize=13, fontweight="bold")
    plt.tight_layout()
    guardar(fig, "09_optuna_convergencia.png")


# ---------------------------------------------------------------------------
# B10. IMPORTANCIA DE HIPERPARÁMETROS (Optuna FAnova)
# ---------------------------------------------------------------------------

def grafico_importancia_hiperparametros(estudios: dict):
    try:
        from optuna.importance import get_param_importances
    except ImportError:
        print("  [SKIP] optuna.importance no disponible.")
        return

    n     = len(estudios)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = axes.flatten()

    for i, (nombre, study) in enumerate(estudios.items()):
        ax = axes[i]
        try:
            importancias = get_param_importances(study)
            params_names = list(importancias.keys())
            valores_imp  = list(importancias.values())
            color        = COLORES.get(nombre, "steelblue")

            bars = ax.barh(params_names, valores_imp, color=color, alpha=0.8,
                           edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, valores_imp):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=8)

            ax.set_title(nombre, fontsize=10, fontweight="bold")
            ax.set_xlabel("Importancia relativa (FAnova)", fontsize=9)
            ax.set_xlim(0, max(valores_imp) * 1.25)
            ax.grid(axis="x", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
        except Exception as e:
            ax.set_title(f"{nombre}\n(no disponible)", fontsize=9)
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Importancia de Hiperparámetros — Optuna FAnova",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    guardar(fig, "10_optuna_hiperparametros.png")


# ---------------------------------------------------------------------------
# C11. FEATURE IMPORTANCE NATIVA — MODELOS DE ÁRBOL
# ---------------------------------------------------------------------------

def grafico_feature_importance_nativa(modelos: dict, features: list):
    arbol_modelos = {n: d for n, d in modelos.items() if n in MODELOS_ARBOL}
    if not arbol_modelos:
        print("  [SKIP] Sin modelos de árbol para feature importance.")
        return

    n     = len(arbol_modelos)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    top_n = 20

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 6))
    axes = axes.flatten()

    for i, (nombre, datos) in enumerate(arbol_modelos.items()):
        model = datos["model"]
        ax    = axes[i]

        try:
            importancias = model.feature_importances_
        except AttributeError:
            ax.set_title(f"{nombre}\n(no soportado)", fontsize=9)
            continue

        idx_top = np.argsort(importancias)[-top_n:]
        feat_top = [features[j] for j in idx_top]
        vals_top = importancias[idx_top]

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(feat_top)))
        bars   = ax.barh(feat_top, vals_top, color=colors, edgecolor="white", linewidth=0.4)

        for bar, val in zip(bars, vals_top):
            ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=7.5)

        ax.set_title(f"{nombre} — Top {top_n} Features", fontsize=10, fontweight="bold")
        ax.set_xlabel("Importancia (MDI)", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Importance Nativa — Modelos de Árbol",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    guardar(fig, "11_feature_importance_nativa.png")


# ---------------------------------------------------------------------------
# C12. TOP-15 FEATURE IMPORTANCE COMPARATIVO
# ---------------------------------------------------------------------------

def grafico_feature_importance_comparativo(modelos: dict, features: list):
    arbol_modelos = {n: d for n, d in modelos.items() if n in MODELOS_ARBOL}
    if len(arbol_modelos) < 2:
        print("  [SKIP] Se necesitan al menos 2 modelos de árbol para el comparativo.")
        return

    top_n = 15
    # Construir DataFrame de importancias normalizadas
    df_imp = pd.DataFrame(index=features)

    for nombre, datos in arbol_modelos.items():
        model = datos["model"]
        try:
            imp = model.feature_importances_
            imp_norm = imp / imp.sum()
            df_imp[nombre] = imp_norm
        except AttributeError:
            pass

    df_imp = df_imp.dropna(axis=1, how="all")
    if df_imp.empty:
        print("  [SKIP] No se pudieron extraer importancias.")
        return

    # Media de importancias para ordenar
    df_imp["media"] = df_imp.mean(axis=1)
    top_features    = df_imp.nlargest(top_n, "media").index.tolist()
    df_plot         = df_imp.loc[top_features].drop(columns="media")

    fig, ax = plt.subplots(figsize=(10, 7))
    x      = np.arange(len(top_features))
    ancho  = 0.8 / len(df_plot.columns)

    for i, col in enumerate(df_plot.columns):
        offsets = x + i * ancho - (len(df_plot.columns) - 1) * ancho / 2
        ax.bar(offsets, df_plot[col].values, ancho,
               label=col, color=COLORES.get(col, f"C{i}"),
               alpha=0.85, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Importancia normalizada", fontsize=11)
    ax.set_title(f"Top {top_n} Features — Comparativo entre Modelos",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    guardar(fig, "12_feature_importance_comparativo.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    inicio = datetime.now()
    print("=" * 80)
    print("SCRIPT 06 — GRÁFICOS DE PERFORMANCE Y FEATURE IMPORTANCE")
    print(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    os.makedirs(GRAFICOS_PATH, exist_ok=True)

    # Verificar dependencias
    archivos_requeridos = [
        os.path.join(REPORTES_PATH, "resultados_optuna.json"),
        os.path.join(REPORTES_PATH, "modelos_optuna.pkl"),
        os.path.join(REPORTES_PATH, "estudios_optuna.pkl"),
        os.path.join(MASTER_PATH,   "X_val.csv"),
        os.path.join(MASTER_PATH,   "y_val.csv"),
        os.path.join(MASTER_PATH,   "X_backtest.csv"),
        os.path.join(MASTER_PATH,   "y_backtest.csv"),
    ]
    for ruta in archivos_requeridos:
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró {ruta}. Ejecuta primero Script 05.")

    print("\n[1] Cargando datos y modelos ...")
    (X_val, y_val, X_backtest, y_backtest,
     resultados, modelos, estudios,
     nombres, ganador, features) = cargar_todo()

    # ── A. Performance ──────────────────────────────────────────────────────
    print("\n[2] Generando gráficos de performance ...")

    grafico_comparacion_metricas(resultados)

    grafico_roc(modelos, y_val,      "Validación", "02_roc_val.png")
    grafico_roc(modelos, y_backtest, "Backtest",   "03_roc_backtest.png")

    grafico_pr(modelos, y_val,      "Validación", "04_pr_val.png")
    grafico_pr(modelos, y_backtest, "Backtest",   "05_pr_backtest.png")

    grafico_confusion(modelos, y_val,      "Validación", "06_confusion_val.png")
    grafico_confusion(modelos, y_backtest, "Backtest",   "07_confusion_backtest.png")

    grafico_dist_probabilidades(modelos, ganador, y_val, y_backtest)

    # ── B. Optuna ───────────────────────────────────────────────────────────
    print("\n[3] Generando gráficos de Optuna ...")
    grafico_convergencia(estudios)
    grafico_importancia_hiperparametros(estudios)

    # ── C. Feature importance ───────────────────────────────────────────────
    print("\n[4] Generando gráficos de feature importance ...")
    grafico_feature_importance_nativa(modelos, features)
    grafico_feature_importance_comparativo(modelos, features)

    # ── Estado ──────────────────────────────────────────────────────────────
    duracion = round((datetime.now() - inicio).total_seconds(), 2)
    graficos_generados = [
        "01_comparacion_metricas.png",
        "02_roc_val.png",
        "03_roc_backtest.png",
        "04_pr_val.png",
        "05_pr_backtest.png",
        "06_confusion_val.png",
        "07_confusion_backtest.png",
        "08_dist_probabilidades.png",
        "09_optuna_convergencia.png",
        "10_optuna_hiperparametros.png",
        "11_feature_importance_nativa.png",
        "12_feature_importance_comparativo.png",
    ]
    estado = {
        "script"             : "06_graficos",
        "timestamp"          : datetime.now().isoformat(),
        "duracion_seg"       : duracion,
        "modelo_ganador"     : ganador,
        "graficos_generados" : graficos_generados,
        "directorio_salida"  : GRAFICOS_PATH,
    }
    os.makedirs(REPORTES_PATH, exist_ok=True)
    with open(os.path.join(REPORTES_PATH, "estado_graficos.json"), "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, ensure_ascii=False)

    print(f"\n  {len(graficos_generados)} gráficos guardados en '{GRAFICOS_PATH}/'")
    print(f"  Duración total : {duracion}s")
    print("Script 06 completado.\n")


if __name__ == "__main__":
    main()
