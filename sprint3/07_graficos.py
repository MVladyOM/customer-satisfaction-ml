from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

ROOT     = Path(__file__).resolve().parent.parent
REPORTES = ROOT / "data" / "reportes"
REPORTS  = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
DATA     = ROOT / "data" / "master"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, nombre: str):
    path = REPORTS / nombre
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   → reports/{nombre}")
    return nombre


def _cargar_datos():
    X_val  = pd.read_csv(DATA / "X_val.csv")
    X_bt   = pd.read_csv(DATA / "X_backtest.csv")
    y_val  = pd.read_csv(DATA / "y_val.csv").squeeze()
    y_bt   = pd.read_csv(DATA / "y_backtest.csv").squeeze()
    return X_val, X_bt, y_val, y_bt


def _cargar_artefactos():
    modelos    = joblib.load(REPORTES / "modelos_optuna.pkl")
    estudios   = joblib.load(REPORTES / "estudios_optuna.pkl")
    resultados = json.loads((REPORTES / "resultados_optuna.json").read_text())
    baseline   = json.loads((REPORTES / "resultados_baseline.json").read_text())
    return modelos, estudios, resultados, baseline


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Comparación de métricas baseline vs optuna
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparacion_metricas(baseline, resultados):
    nombres_base = [r["modelo"] for r in baseline]
    nombres_opt  = [r["modelo"] for r in resultados]
    nombres      = sorted(set(nombres_base) & set(nombres_opt))

    auc_base = {r["modelo"]: r.get("auc_val", r.get("AUC-ROC", 0)) for r in baseline}
    auc_opt  = {r["modelo"]: r["auc_val"] for r in resultados}

    x = np.arange(len(nombres))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, [auc_base.get(n, 0) for n in nombres], w, label="Baseline", color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + w/2, [auc_opt.get(n, 0) for n in nombres],  w, label="Optuna",   color="#DD8452", alpha=0.85)
    ax.bar_label(b1, fmt="%.4f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.4f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUC-ROC (Validación)")
    ax.set_title("Comparación AUC-ROC — Baseline vs Optuna", fontsize=12)
    ax.legend()
    ax.set_ylim(0.5, ax.get_ylim()[1] + 0.03)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, "01_comparacion_metricas.png")


# ─────────────────────────────────────────────────────────────────────────────
# 02 — Curvas ROC en validación (todos los modelos)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_val(modelos, y_val, X_val):
    fig, ax = plt.subplots(figsize=(8, 6))
    for nombre, modelo in sorted(modelos.items(),
                                  key=lambda kv: roc_auc_score(y_val, kv[1].predict_proba(X_val)[:, 1]),
                                  reverse=True):
        proba = modelo.predict_proba(X_val)[:, 1]
        RocCurveDisplay.from_predictions(
            y_val, proba,
            name=f"{nombre} ({roc_auc_score(y_val, proba):.4f})",
            ax=ax,
        )
    ax.plot([0,1],[0,1],"k--", label="Random (0.5000)")
    ax.set_title("Curvas ROC — Validación", fontsize=12)
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "02_roc_val.png")


# ─────────────────────────────────────────────────────────────────────────────
# 03 — Curvas ROC en backtest
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_bt(modelos, y_bt, X_bt):
    fig, ax = plt.subplots(figsize=(8, 6))
    for nombre, modelo in sorted(modelos.items(),
                                  key=lambda kv: roc_auc_score(y_bt, kv[1].predict_proba(X_bt)[:, 1]),
                                  reverse=True):
        proba = modelo.predict_proba(X_bt)[:, 1]
        RocCurveDisplay.from_predictions(
            y_bt, proba,
            name=f"{nombre} ({roc_auc_score(y_bt, proba):.4f})",
            ax=ax,
        )
    ax.plot([0,1],[0,1],"k--", label="Random (0.5000)")
    ax.set_title("Curvas ROC — Backtest", fontsize=12)
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "03_roc_backtest.png")


# ─────────────────────────────────────────────────────────────────────────────
# 04 — Curvas Precision-Recall en validación
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_val(modelos, y_val, X_val):
    fig, ax = plt.subplots(figsize=(8, 6))
    prevalencia = float(y_val.mean())
    for nombre, modelo in modelos.items():
        proba = modelo.predict_proba(X_val)[:, 1]
        ap = average_precision_score(y_val, proba)
        PrecisionRecallDisplay.from_predictions(
            y_val, proba,
            name=f"{nombre} (AP={ap:.4f})",
            ax=ax,
        )
    ax.axhline(prevalencia, color="gray", ls="--", label=f"Baseline prevalencia={prevalencia:.2f}")
    ax.set_title("Curvas Precision-Recall — Validación", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "04_pr_val.png")


# ─────────────────────────────────────────────────────────────────────────────
# 05 — Curvas Precision-Recall en backtest
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_bt(modelos, y_bt, X_bt):
    fig, ax = plt.subplots(figsize=(8, 6))
    prevalencia = float(y_bt.mean())
    for nombre, modelo in modelos.items():
        proba = modelo.predict_proba(X_bt)[:, 1]
        ap = average_precision_score(y_bt, proba)
        PrecisionRecallDisplay.from_predictions(
            y_bt, proba,
            name=f"{nombre} (AP={ap:.4f})",
            ax=ax,
        )
    ax.axhline(prevalencia, color="gray", ls="--", label=f"Baseline prevalencia={prevalencia:.2f}")
    ax.set_title("Curvas Precision-Recall — Backtest", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "05_pr_backtest.png")


# ─────────────────────────────────────────────────────────────────────────────
# 06 — Matrices de confusión en validación (grilla)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_val(modelos, y_val, X_val):
    n = len(modelos)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()
    for i, (nombre, modelo) in enumerate(modelos.items()):
        pred = (modelo.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        cm = confusion_matrix(y_val, pred)
        ConfusionMatrixDisplay(cm, display_labels=["Insatisfecho", "Satisfecho"]).plot(
            ax=axes[i], colorbar=False, cmap="Blues"
        )
        axes[i].set_title(nombre, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Matrices de Confusión — Validación", fontsize=12)
    plt.tight_layout()
    return _save(fig, "06_confusion_val.png")


# ─────────────────────────────────────────────────────────────────────────────
# 07 — Matrices de confusión en backtest
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_bt(modelos, y_bt, X_bt):
    n = len(modelos)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = axes.flatten()
    for i, (nombre, modelo) in enumerate(modelos.items()):
        pred = (modelo.predict_proba(X_bt)[:, 1] >= 0.5).astype(int)
        cm = confusion_matrix(y_bt, pred)
        ConfusionMatrixDisplay(cm, display_labels=["Insatisfecho", "Satisfecho"]).plot(
            ax=axes[i], colorbar=False, cmap="Oranges"
        )
        axes[i].set_title(nombre, fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Matrices de Confusión — Backtest", fontsize=12)
    plt.tight_layout()
    return _save(fig, "07_confusion_backtest.png")


# ─────────────────────────────────────────────────────────────────────────────
# 08 — Distribución de probabilidades predichas (mejor modelo)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dist_probabilidades(modelos, resultados, y_val, X_val):
    ganador_nombre = max(resultados, key=lambda r: r["f1_0_val"])["modelo"]
    modelo = modelos.get(ganador_nombre)
    if modelo is None:
        modelo = list(modelos.values())[0]
        ganador_nombre = list(modelos.keys())[0]

    proba = modelo.predict_proba(X_val)[:, 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    for clase, color, label in [(0, "#e74c3c", "Insatisfecho (cls 0)"),
                                 (1, "#2ecc71", "Satisfecho (cls 1)")]:
        mask = y_val == clase
        ax.hist(proba[mask], bins=50, alpha=0.6, color=color, label=label, density=True)
    ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Umbral 0.5")
    ax.set_xlabel("Probabilidad predicha (clase 1)")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de Probabilidades — {ganador_nombre}\n(Validación)", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "08_dist_probabilidades.png")


# ─────────────────────────────────────────────────────────────────────────────
# 09 — Convergencia de Optuna por modelo
# ─────────────────────────────────────────────────────────────────────────────

def plot_optuna_convergencia(estudios):
    n = len(estudios)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for i, (nombre, study) in enumerate(estudios.items()):
        valores = [t.value for t in study.trials if t.value is not None]
        mejor_acum = np.maximum.accumulate(valores)
        ax = axes[i]
        ax.plot(valores, alpha=0.4, color="steelblue", label="Trial")
        ax.plot(mejor_acum, color="red", lw=2, label="Mejor acum.")
        ax.set_title(nombre, fontsize=9)
        ax.set_xlabel("Trial", fontsize=8)
        ax.set_ylabel("F1 cls 0", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Convergencia de Optuna por modelo", fontsize=12, y=1.01)
    plt.tight_layout()
    return _save(fig, "09_optuna_convergencia.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10 — Importancia de hiperparámetros (Optuna)
# ─────────────────────────────────────────────────────────────────────────────

def plot_optuna_hiperparametros(estudios):
    from optuna.importance import get_param_importances

    n = len(estudios)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    for i, (nombre, study) in enumerate(estudios.items()):
        ax = axes[i]
        try:
            imp = get_param_importances(study)
            params_n = list(imp.keys())
            valores  = list(imp.values())
            ax.barh(params_n, valores, color="steelblue")
            ax.set_title(nombre, fontsize=9)
            ax.set_xlabel("Importancia relativa", fontsize=8)
            ax.grid(axis="x", alpha=0.3)
        except Exception as e:
            ax.set_title(f"{nombre}\n(no disponible)", fontsize=9)
            ax.axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Importancia de hiperparámetros — Optuna", fontsize=12, y=1.01)
    plt.tight_layout()
    return _save(fig, "10_optuna_hiperparametros.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11 — Feature importance nativa del mejor modelo
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance_nativa(modelos, resultados, X_val):
    ganador_nombre = max(resultados, key=lambda r: r["f1_0_val"])["modelo"]
    modelo = modelos.get(ganador_nombre)
    features = X_val.columns.tolist()

    importancias = None
    label = ""
    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
        label = "Importancia (gain)"
    elif hasattr(modelo, "coef_"):
        importancias = np.abs(modelo.coef_[0])
        label = "|Coeficiente|"
    elif hasattr(modelo, "estimators_"):
        # CalibratedClassifierCV wrapping LinearSVC
        for est in modelo.estimators_:
            if hasattr(est, "coef_"):
                importancias = np.abs(est.coef_[0])
                label = "|Coeficiente| (calibrado)"
                break

    if importancias is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"Feature importance no disponible\npara {ganador_nombre}",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.axis("off")
        return _save(fig, "11_feature_importance_nativa.png")

    df_imp = pd.DataFrame({"Feature": features, "Importancia": importancias})
    df_imp = df_imp.sort_values("Importancia", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df_imp["Feature"], df_imp["Importancia"], color="steelblue")
    ax.set_xlabel(label)
    ax.set_title(f"Top 20 Features — {ganador_nombre}", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return _save(fig, "11_feature_importance_nativa.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12 — Feature importance comparativa (modelos con importancias)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance_comparativo(modelos, X_val):
    features = X_val.columns.tolist()

    candidatos = {}
    for nombre, modelo in modelos.items():
        if hasattr(modelo, "feature_importances_"):
            candidatos[nombre] = modelo.feature_importances_
        elif hasattr(modelo, "coef_"):
            candidatos[nombre] = np.abs(modelo.coef_[0])

    if not candidatos:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Ningún modelo expone feature importance directamente",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.axis("off")
        return _save(fig, "12_feature_importance_comparativo.png")

    n = len(candidatos)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, (nombre, imp) in enumerate(candidatos.items()):
        df_imp = pd.DataFrame({"Feature": features, "Imp": imp})
        df_imp = df_imp.sort_values("Imp", ascending=True).tail(15)
        axes[i].barh(df_imp["Feature"], df_imp["Imp"], color="darkorange", alpha=0.8)
        axes[i].set_title(nombre, fontsize=10)
        axes[i].set_xlabel("Importancia", fontsize=8)
        axes[i].grid(axis="x", alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Importance comparativa — Top 15 por modelo", fontsize=12)
    plt.tight_layout()
    return _save(fig, "12_feature_importance_comparativo.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_inicio = time.time()
    print("=" * 55)
    print("07_graficos.py — Gráficos de performance Sprint 3")
    print("=" * 55)

    X_val, X_bt, y_val, y_bt = _cargar_datos()
    modelos, estudios, resultados, baseline = _cargar_artefactos()
    print(f"\n{len(modelos)} modelos cargados\n")

    generados = []
    print("Generando gráficos...")
    generados.append(plot_comparacion_metricas(baseline, resultados))
    generados.append(plot_roc_val(modelos, y_val, X_val))
    generados.append(plot_roc_bt(modelos, y_bt, X_bt))
    generados.append(plot_pr_val(modelos, y_val, X_val))
    generados.append(plot_pr_bt(modelos, y_bt, X_bt))
    generados.append(plot_confusion_val(modelos, y_val, X_val))
    generados.append(plot_confusion_bt(modelos, y_bt, X_bt))
    generados.append(plot_dist_probabilidades(modelos, resultados, y_val, X_val))
    generados.append(plot_optuna_convergencia(estudios))
    generados.append(plot_optuna_hiperparametros(estudios))
    generados.append(plot_feature_importance_nativa(modelos, resultados, X_val))
    generados.append(plot_feature_importance_comparativo(modelos, X_val))

    ganador_nombre = max(resultados, key=lambda r: r["f1_0_val"])["modelo"]
    duracion = round(time.time() - t_inicio, 2)

    estado = {
        "script":             "07_graficos",
        "timestamp":          datetime.now().isoformat(),
        "duracion_seg":       duracion,
        "modelo_ganador":     ganador_nombre,
        "graficos_generados": generados,
        "directorio_salida":  "reports",
    }
    with open(REPORTES / "estado_graficos.json", "w") as f:
        json.dump(estado, f, indent=2)

    print(f"\n✅ 07_graficos.py completado en {duracion}s  ({len(generados)} gráficos)")
    print(f"   → data/reportes/estado_graficos.json")


if __name__ == "__main__":
    main()