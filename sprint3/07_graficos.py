from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
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

TOP3 = ["Hist Gradient Boosting", "Logistic Regression", "LinearSVM"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, nombre: str):
    path = REPORTS / nombre
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> reports/{nombre}")
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
    pkl_base   = REPORTES / "modelos_baseline.pkl"
    modelos_baseline = joblib.load(pkl_base) if pkl_base.exists() else {}
    return modelos, estudios, resultados, baseline, modelos_baseline


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
# 13 — Importancia de variables: Baseline vs Optuna
# ─────────────────────────────────────────────────────────────────────────────

def plot_importancia_antes_despues(modelos_baseline, modelos_optuna, X_val):
    """
    Para cada modelo que expone feature_importances_ en ambas versiones
    (baseline y optuna), muestra un gráfico de barras side-by-side con las
    15 features más importantes según la versión optuna.
    """
    features = X_val.columns.tolist()

    # Mapa de nombres baseline -> optuna (pueden diferir ligeramente)
    nombre_map = {
        "Random Forest":   "Random Forest",
        "ExtraTrees":      "ExtraTrees",
        "HGBC":            "Hist Gradient Boosting",
        "CatBoost":        "CatBoost",
        "LightGBM":        "LightGBM",
    }

    pares = []
    for nb, no in nombre_map.items():
        mb = modelos_baseline.get(nb)
        mo = modelos_optuna.get(no)
        if mb is None or mo is None:
            continue
        if not (hasattr(mb, "feature_importances_") and hasattr(mo, "feature_importances_")):
            continue
        pares.append((no, mb.feature_importances_, mo.feature_importances_))

    if not pares:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5,
                "No hay modelos con feature_importances_ en ambas versiones.\n"
                "Ejecuta primero 05_baseline.py para generar modelos_baseline.pkl",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.axis("off")
        return _save(fig, "13_importancia_antes_despues.png")

    n = len(pares)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 9), squeeze=False)

    for col, (nombre, imp_base, imp_opt) in enumerate(pares):
        ax = axes[0][col]

        # Ordenar por importancia optuna (desc) y tomar top 15
        df = pd.DataFrame({
            "Feature":   features,
            "Baseline":  imp_base,
            "Optuna":    imp_opt,
        }).nlargest(15, "Optuna").sort_values("Optuna")

        y     = np.arange(len(df))
        alto  = 0.35

        b1 = ax.barh(y - alto / 2, df["Baseline"], alto,
                     label="Baseline", color="#4C72B0", alpha=0.85)
        b2 = ax.barh(y + alto / 2, df["Optuna"],   alto,
                     label="Optuna",   color="#DD8452", alpha=0.85)

        ax.set_yticks(y)
        ax.set_yticklabels(df["Feature"], fontsize=8)
        ax.set_xlabel("Importancia (MDI)", fontsize=9)
        ax.set_title(f"{nombre}\nBaseline vs Optuna — Top 15 features",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        # Anotar diferencia porcentual en cada barra optuna
        for bar, vb, vo in zip(b2, df["Baseline"], df["Optuna"]):
            if vb > 0:
                delta = (vo - vb) / vb * 100
                ax.text(bar.get_width() + 0.001,
                        bar.get_y() + bar.get_height() / 2,
                        f"{delta:+.0f}%", va="center", ha="left",
                        fontsize=6.5,
                        color="#c0392b" if delta < 0 else "#27ae60")

    plt.suptitle("Importancia de variables — Antes (Baseline) vs Después (Optuna)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save(fig, "13_importancia_antes_despues.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 14 — Métricas de desbalanceo (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metricas_desbalanceo(baseline):
    nombres = [r["modelo"] for r in baseline]

    mcc      = [r.get("mcc_val",      0) for r in baseline]
    kappa    = [r.get("kappa_val",    0) for r in baseline]
    balacc   = [r.get("bal_acc_val",  0) for r in baseline]
    gmean    = [r.get("gmean_val",    0) for r in baseline]
    auc      = [r.get("auc_val",      0) for r in baseline]
    avgprec  = [r.get("avg_prec_val", 0) for r in baseline]

    # Ordenar por MCC descendente
    orden = sorted(range(len(nombres)), key=lambda i: mcc[i], reverse=True)
    nom_o  = [nombres[i] for i in orden]
    mcc_o  = [mcc[i]     for i in orden]
    kap_o  = [kappa[i]   for i in orden]
    bal_o  = [balacc[i]  for i in orden]
    gm_o   = [gmean[i]   for i in orden]
    auc_o  = [auc[i]     for i in orden]
    ap_o   = [avgprec[i] for i in orden]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Subplot izquierdo: MCC / Kappa / BalAcc / Gmean ─────────────────────
    x  = np.arange(len(nom_o))
    w  = 0.2
    cols = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    labels = ["MCC", "Kappa", "Bal. Accuracy", "G-mean"]
    for i, (vals, lbl, col) in enumerate(zip([mcc_o, kap_o, bal_o, gm_o], labels, cols)):
        bars = ax1.bar(x + (i - 1.5) * w, vals, w, label=lbl, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax1.set_xticks(x)
    ax1.set_xticklabels(nom_o, rotation=20, ha="right", fontsize=9)
    ax1.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax1.set_ylim(-0.05, max(max(bal_o), max(gm_o)) + 0.12)
    ax1.set_ylabel("Valor de la métrica")
    ax1.set_title("Métricas de desbalanceo — Baseline\n(MCC, Kappa, Balanced Accuracy, G-mean)",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Subplot derecho: AUC-ROC vs Average Precision ────────────────────────
    x2 = np.arange(len(nom_o))
    b1 = ax2.bar(x2 - 0.22, auc_o, 0.4, label="AUC-ROC",   color="#4C72B0", alpha=0.85)
    b2 = ax2.bar(x2 + 0.22, ap_o,  0.4, label="Avg Prec (PR)", color="#DD8452", alpha=0.85)
    for bar, v in zip(list(b1) + list(b2), auc_o + ap_o):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # Flecha que marca la brecha entre AUC y AvgPrec para el peor caso
    for i, (a, p) in enumerate(zip(auc_o, ap_o)):
        if p < a - 0.04:
            ax2.annotate("", xy=(x2[i] + 0.22, p + 0.01),
                         xytext=(x2[i] - 0.22, a - 0.01),
                         arrowprops=dict(arrowstyle="-|>", color="gray", lw=0.8))

    ax2.set_xticks(x2)
    ax2.set_xticklabels(nom_o, rotation=20, ha="right", fontsize=9)
    ax2.set_ylim(0.5, 1.05)
    ax2.set_ylabel("Score")
    ax2.set_title("AUC-ROC vs Average Precision (PR)\n"
                  "(brecha mayor indica mayor impacto del desbalanceo)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Análisis de desbalanceo de clases — Baseline (Validación)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "14_metricas_desbalanceo.png")


# ─────────────────────────────────────────────────────────────────────────────
# 15 — Detección de overfitting (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def plot_overfitting(baseline):
    UMBRAL = 0.05

    nombres    = [r["modelo"]            for r in baseline]
    auc_val    = [r.get("auc_val",       0) for r in baseline]
    auc_tr     = [r.get("auc_train",     0) for r in baseline]
    f1_val     = [r.get("f1_macro_val",  0) for r in baseline]
    f1_tr      = [r.get("f1_macro_train",0) for r in baseline]
    delta_auc  = [r.get("delta_auc",  0) for r in baseline]
    delta_mcc  = [r.get("delta_mcc",  0) for r in baseline]
    overfit    = [r.get("overfit",    False) for r in baseline]

    # Ordenar por delta_auc descendente (peor overfitting primero)
    orden = sorted(range(len(nombres)), key=lambda i: delta_auc[i], reverse=True)
    nom_o   = [nombres[i]   for i in orden]
    av_o    = [auc_val[i]   for i in orden]
    at_o    = [auc_tr[i]    for i in orden]
    fv_o    = [f1_val[i]    for i in orden]
    ft_o    = [f1_tr[i]     for i in orden]
    da_o    = [delta_auc[i] for i in orden]
    dm_o    = [delta_mcc[i] for i in orden]
    ov_o    = [overfit[i]   for i in orden]

    fig, axes = plt.subplots(3, 1, figsize=(13, 14))
    x = np.arange(len(nom_o))
    w = 0.35

    # ── Subplot 1: AUC — Train vs Val ────────────────────────────────────────
    ax = axes[0]
    b_tr  = ax.bar(x - w/2, at_o, w, label="AUC Train", color="#2980b9", alpha=0.85)
    b_val = ax.bar(x + w/2, av_o, w, label="AUC Val",   color="#e74c3c", alpha=0.85)
    for bar, v in zip(list(b_tr) + list(b_val), at_o + av_o):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    # Anotar delta
    for i, (a, b, ov) in enumerate(zip(at_o, av_o, ov_o)):
        d = a - b
        color = "#c0392b" if ov else "#27ae60"
        ax.annotate(f"Δ={d:+.3f}", xy=(x[i], max(a, b) + 0.018),
                    ha="center", fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(nom_o, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.5, max(max(at_o), 1.0) + 0.08)
    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6, label="AUC=1 (memorización)")
    ax.set_ylabel("AUC-ROC"); ax.set_title("AUC-ROC — Train vs Validación", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Subplot 2: F1_macro — Train vs Val ───────────────────────────────────
    ax = axes[1]
    b_tr  = ax.bar(x - w/2, ft_o, w, label="F1_macro Train", color="#8e44ad", alpha=0.85)
    b_val = ax.bar(x + w/2, fv_o, w, label="F1_macro Val",   color="#d35400", alpha=0.85)
    for bar, v in zip(list(b_tr) + list(b_val), ft_o + fv_o):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    for i, (a, b, ov) in enumerate(zip(ft_o, fv_o, ov_o)):
        d = a - b
        color = "#c0392b" if ov else "#27ae60"
        ax.annotate(f"Δ={d:+.3f}", xy=(x[i], max(a, b) + 0.015),
                    ha="center", fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(nom_o, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.4, max(max(ft_o), 1.0) + 0.08)
    ax.set_ylabel("F1 Macro"); ax.set_title("F1 Macro — Train vs Validación", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3, linestyle="--")

    # ── Subplot 3: Deltas con línea de umbral ────────────────────────────────
    ax = axes[2]
    colores_da = ["#c0392b" if ov else "#2980b9" for ov in ov_o]
    colores_df = ["#c0392b" if ov else "#27ae60"  for ov in ov_o]

    b1 = ax.bar(x - w/2, da_o, w, label="Δ AUC (tr-val)",     color=colores_da, alpha=0.85)
    b2 = ax.bar(x + w/2, dm_o, w, label="Δ MCC (tr-val)",     color=colores_df, alpha=0.85)
    for bar, v in zip(list(b1) + list(b2), da_o + dm_o):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.005 if v >= 0 else -0.015),
                f"{v:+.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(UMBRAL,  color="#e74c3c", ls="--", lw=1.5, label=f"Umbral overfit (+{UMBRAL})")
    ax.axhline(-UMBRAL, color="#3498db", ls="--", lw=1.5, label=f"Umbral underfit (-{UMBRAL})")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)

    # Etiquetas de overfit
    for i, ov in enumerate(ov_o):
        if ov:
            ax.text(x[i], max(da_o[i], dm_o[i]) + 0.03, "OVERFIT",
                    ha="center", fontsize=7, color="#c0392b", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(nom_o, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Delta (Train - Val)")
    ax.set_title("Diferencia Train-Val por modelo\n"
                 "(rojo = supera umbral de overfitting)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.suptitle("Detección de Overfitting — Baseline",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save(fig, "15_overfitting.png")


# ─────────────────────────────────────────────────────────────────────────────
# 16 — Tabla de métricas Optuna — Top 3 modelos
# ─────────────────────────────────────────────────────────────────────────────

def plot_tabla_metricas_top3(resultados):
    datos = {r["modelo"]: r for r in resultados if r["modelo"] in TOP3}
    modelos_presentes = [m for m in TOP3 if m in datos]

    if not modelos_presentes:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No se encontraron los modelos top-3 en resultados_optuna.json",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _save(fig, "16_tabla_metricas_top3.png")

    def fmt(r, key):
        if key == "cv":
            return f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}"
        return f"{r[key]:.4f}"

    FILAS = [
        # (etiqueta,            clave,            seccion)
        ("── Validación ──────────────", None,            "sep_val"),
        ("CV F1 (mean ± std)",           "cv",            "cv"),
        ("F1  Cls 0",                    "f1_0_val",      "val"),
        ("Recall  Cls 0",                "recall_0_val",  "val"),
        ("Prec  Cls 0",                  "prec_0_val",    "val"),
        ("F1  Cls 1",                    "f1_1_val",      "val"),
        ("Recall  Cls 1",                "recall_1_val",  "val"),
        ("Prec  Cls 1",                  "prec_1_val",    "val"),
        ("F1  Macro",                    "f1_macro_val",  "val"),
        ("AUC-ROC",                      "auc_val",       "val"),
        ("Accuracy",                     "accuracy_val",  "val"),
        ("── Backtest ────────────────", None,            "sep_bt"),
        ("F1  Cls 0",                    "f1_0_bt",       "bt"),
        ("Recall  Cls 0",                "recall_0_bt",   "bt"),
        ("Prec  Cls 0",                  "prec_0_bt",     "bt"),
        ("F1  Cls 1",                    "f1_1_bt",       "bt"),
        ("Recall  Cls 1",                "recall_1_bt",   "bt"),
        ("Prec  Cls 1",                  "prec_1_bt",     "bt"),
        ("F1  Macro",                    "f1_macro_bt",   "bt"),
        ("AUC-ROC",                      "auc_bt",        "bt"),
        ("Accuracy",                     "accuracy_bt",   "bt"),
    ]

    col_labels = ["Métrica"] + modelos_presentes
    table_data = []
    for etiqueta, clave, seccion in FILAS:
        if seccion.startswith("sep"):
            row = [etiqueta] + [""] * len(modelos_presentes)
        else:
            row = [etiqueta] + [fmt(datos[m], clave) for m in modelos_presentes]
        table_data.append(row)

    n_cols = len(col_labels)
    fig_h  = max(8, 0.45 * len(FILAS) + 1.5)
    fig, ax = plt.subplots(figsize=(4 + 3.2 * len(modelos_presentes), fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Encabezado de columnas
    for c in range(n_cols):
        cell = tbl[0, c]
        cell.set_facecolor("#1A3A5C")
        cell.set_text_props(color="white", fontweight="bold")

    val_colors = ["#EBF5FB", "#FAFAFA"]
    bt_colors  = ["#FEF9E7", "#FAFAFA"]
    val_i = bt_i = 0

    for r_idx, (_, _, seccion) in enumerate(FILAS):
        tbl_row = r_idx + 1
        is_sep = seccion.startswith("sep")

        if is_sep:
            bg = "#2C3E50"
        elif seccion == "cv":
            bg = "#EAFAF1"
        elif seccion == "val":
            bg = val_colors[val_i % 2]; val_i += 1
        else:
            bg = bt_colors[bt_i % 2]; bt_i += 1

        for c in range(n_cols):
            cell = tbl[tbl_row, c]
            cell.set_facecolor(bg)
            if is_sep:
                cell.set_text_props(color="white", fontsize=8.5, fontweight="bold")

        tbl[tbl_row, 0].set_text_props(fontweight="bold")

    ax.set_title(
        "Métricas Optuna — Top 3 Modelos\n"
        "(Hist Gradient Boosting · Logistic Regression · LinearSVM)",
        fontsize=12, fontweight="bold", pad=16,
    )
    plt.tight_layout()
    return _save(fig, "16_tabla_metricas_top3.png")


def main():
    t_inicio = time.time()
    print("=" * 55)
    print("07_graficos.py — Gráficos de performance Sprint 3")
    print("=" * 55)

    X_val, X_bt, y_val, y_bt = _cargar_datos()
    modelos, estudios, resultados, baseline, modelos_baseline = _cargar_artefactos()
    print(f"\n{len(modelos)} modelos optuna cargados")
    if modelos_baseline:
        print(f"{len(modelos_baseline)} modelos baseline cargados\n")
    else:
        print("modelos_baseline.pkl no encontrado — ejecuta 05_baseline.py primero\n")

    # Filtrar a los 3 modelos seleccionados para todos los plots de Optuna
    modelos_top3    = {k: v for k, v in modelos.items()  if k in TOP3}
    resultados_top3 = [r for r in resultados              if r["modelo"] in TOP3]
    estudios_top3   = {k: v for k, v in estudios.items() if k in TOP3}
    print(f"Modelos filtrados (top 3): {list(modelos_top3.keys())}\n")

    generados = []
    print("Generando graficos...")
    generados.append(plot_comparacion_metricas(baseline, resultados_top3))
    generados.append(plot_roc_val(modelos_top3, y_val, X_val))
    generados.append(plot_roc_bt(modelos_top3, y_bt, X_bt))
    generados.append(plot_pr_val(modelos_top3, y_val, X_val))
    generados.append(plot_pr_bt(modelos_top3, y_bt, X_bt))
    generados.append(plot_confusion_val(modelos_top3, y_val, X_val))
    generados.append(plot_confusion_bt(modelos_top3, y_bt, X_bt))
    generados.append(plot_dist_probabilidades(modelos_top3, resultados_top3, y_val, X_val))
    generados.append(plot_optuna_convergencia(estudios_top3))
    generados.append(plot_optuna_hiperparametros(estudios_top3))
    generados.append(plot_feature_importance_nativa(modelos_top3, resultados_top3, X_val))
    generados.append(plot_feature_importance_comparativo(modelos_top3, X_val))
    generados.append(plot_importancia_antes_despues(modelos_baseline, modelos_top3, X_val))
    generados.append(plot_metricas_desbalanceo(baseline))
    generados.append(plot_overfitting(baseline))
    generados.append(plot_tabla_metricas_top3(resultados_top3))

    ganador_nombre = max(resultados_top3, key=lambda r: r["f1_0_val"])["modelo"]
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

    print(f"\n07_graficos.py completado en {duracion}s  ({len(generados)} graficos)")
    print(f"   -> data/reportes/estado_graficos.json")


if __name__ == "__main__":
    main()