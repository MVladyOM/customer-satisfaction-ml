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

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT     = Path(__file__).resolve().parent.parent
REPORTES = ROOT / "data" / "reportes"
REPORTS  = ROOT / "reports" / "sprint4"
REPORTS.mkdir(parents=True, exist_ok=True)
DATA     = ROOT / "data" / "master"

MODELO_ELEGIDO = "Logistic Regression"

PERIODOS_ORDEN = ["Train", "Val", "Backtest"]
ARCHIVOS = {
    "Train":    ("X_train.csv",    "y_train.csv"),
    "Val":      ("X_val.csv",      "y_val.csv"),
    "Backtest": ("X_backtest.csv", "y_backtest.csv"),
}
COLORES_PERIODO = {
    "Train":    "#4C72B0",
    "Val":      "#DD8452",
    "Backtest": "#55A868",
}
COLOR_LR = "#3498db"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _save(fig, nombre: str):
    path = REPORTS / nombre
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> reports/sprint4/{nombre}")
    return nombre


def _metricas(y_true, y_pred, y_proba):
    return {
        "auc":      round(roc_auc_score(y_true, y_proba), 4),
        "avg_prec": round(average_precision_score(y_true, y_proba), 4),
        "prec_0":   round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "recall_0": round(recall_score(y_true, y_pred,    pos_label=0, zero_division=0), 4),
        "prec_1":   round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_1": round(recall_score(y_true, y_pred,    pos_label=1, zero_division=0), 4),
        "f1_0":     round(f1_score(y_true, y_pred,        pos_label=0, zero_division=0), 4),
        "f1_1":     round(f1_score(y_true, y_pred,        pos_label=1, zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "brier":    round(brier_score_loss(y_true, y_proba), 4),
        "accuracy": round(float((y_true == y_pred).mean()), 4),
    }


def _psi(ref_proba: np.ndarray, act_proba: np.ndarray, buckets: int = 10) -> float:
    breakpoints = np.quantile(ref_proba, np.linspace(0, 1, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] = np.inf
    ref_cnt = np.histogram(ref_proba, bins=breakpoints)[0]
    act_cnt = np.histogram(act_proba, bins=breakpoints)[0]
    ref_pct = np.where(ref_cnt == 0, 1e-4, ref_cnt / len(ref_proba))
    act_pct = np.where(act_cnt == 0, 1e-4, act_cnt / len(act_proba))
    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def _umbral_optimo_f1(y_true, y_proba, pos_label: int = 0):
    umbrales = np.linspace(0.05, 0.95, 181)
    f1s = [
        f1_score(y_true, (y_proba >= t).astype(int), pos_label=pos_label, zero_division=0)
        for t in umbrales
    ]
    idx = int(np.argmax(f1s))
    return umbrales[idx], f1s[idx], umbrales, f1s


# -----------------------------------------------------------------------------
# Carga de datos y modelo
# -----------------------------------------------------------------------------

def cargar_datos():
    datos = {}
    for periodo, (fx, fy) in ARCHIVOS.items():
        X = pd.read_csv(DATA / fx)
        y_df = pd.read_csv(DATA / fy)
        y = y_df["satisfecho"] if "satisfecho" in y_df.columns else y_df.squeeze()
        datos[periodo] = (X, y)
        print(f"  {periodo:10s}: {X.shape}  insatisfechos={(y == 0).mean():.2%}")
    return datos


def cargar_modelo():
    pkl = REPORTES / "modelos_optuna_cls0.pkl"
    if not pkl.exists():
        pkl = REPORTES / "modelos_optuna.pkl"
    todos = joblib.load(pkl)
    if MODELO_ELEGIDO not in todos:
        raise KeyError(
            f"Modelo '{MODELO_ELEGIDO}' no encontrado. "
            f"Disponibles: {list(todos.keys())}"
        )
    print(f"  Modelo cargado: {MODELO_ELEGIDO}")
    return todos[MODELO_ELEGIDO]


def calcular_metricas_lr(modelo, datos):
    resultados = {}
    probas = {}
    print(f"  Evaluando {MODELO_ELEGIDO}...")
    for periodo, (X, y) in datos.items():
        proba = modelo.predict_proba(X)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        resultados[periodo] = _metricas(y, pred, proba)
        probas[periodo]     = proba
    return resultados, probas


# -----------------------------------------------------------------------------
# 01 — Dashboard métricas clave: AUC-ROC, Precision, PR-AUC, Recall
# -----------------------------------------------------------------------------

def plot_metricas_clave(resultados):
    # Fila 1: métricas globales + Clase 0 | Fila 2: PR-AUC + Clase 1
    METRICAS_CLAVE = [
        ("auc",      "AUC-ROC",                            "#2980b9"),
        ("prec_0",   "Precision  (Cls 0 — Insatisfecho)",  "#8e44ad"),
        ("recall_0", "Recall  (Cls 0 — Insatisfecho)",     "#e67e22"),
        ("avg_prec", "PR-AUC  (Average Precision)",        "#27ae60"),
        ("prec_1",   "Precision  (Cls 1 — Satisfecho)",    "#6c3483"),
        ("recall_1", "Recall  (Cls 1 — Satisfecho)",       "#d35400"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (clave, titulo, color) in zip(axes, METRICAS_CLAVE):
        vals = [resultados[p][clave] for p in PERIODOS_ORDEN]
        bars = ax.bar(
            PERIODOS_ORDEN, vals,
            color=[COLORES_PERIODO[p] for p in PERIODOS_ORDEN],
            edgecolor="white", linewidth=1.5, width=0.5,
        )
        ax.bar_label(bars, fmt="%.4f", fontsize=11, fontweight="bold", padding=5)
        ax.set_ylim(0, min(1.10, max(vals) + 0.18))
        ax.set_title(titulo, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Valor", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
        ax.tick_params(axis="x", labelsize=11)

    plt.suptitle(
        f"Métricas Clave — {MODELO_ELEGIDO}\nTrain / Val / Backtest",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, "01_metricas_clave_lr.png")


# -----------------------------------------------------------------------------
# 02 — Evolución temporal: Train → Val → Backtest
# -----------------------------------------------------------------------------

def plot_degradacion_temporal(resultados):
    METRICAS_LINEA = [
        ("auc",      "AUC-ROC"),
        ("avg_prec", "PR-AUC"),
        ("prec_0",   "Precision  Cls 0 (Insatisfecho)"),
        ("prec_1",   "Precision  Cls 1 (Satisfecho)"),
        ("recall_0", "Recall  Cls 0 (Insatisfecho)"),
        ("recall_1", "Recall  Cls 1 (Satisfecho)"),
        ("f1_0",     "F1  Clase 0 (Insatisfecho)"),
        ("f1_1",     "F1  Clase 1 (Satisfecho)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    x_vals = list(range(len(PERIODOS_ORDEN)))

    for ax, (metrica, titulo) in zip(axes, METRICAS_LINEA):
        vals = [resultados[p][metrica] for p in PERIODOS_ORDEN]
        ax.plot(x_vals, vals, "o-", color=COLOR_LR, lw=2.5, markersize=11)
        for xi, v in zip(x_vals, vals):
            ax.annotate(
                f"{v:.4f}", (xi, v),
                textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=9, fontweight="bold", color=COLOR_LR,
            )
        ax.fill_between(x_vals, vals, alpha=0.12, color=COLOR_LR)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(PERIODOS_ORDEN, fontsize=11)
        ax.set_ylabel(titulo, fontsize=9)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        lo = max(0, min(vals) - 0.10)
        hi = min(1.05, max(vals) + 0.14)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"Evolución Temporal de Métricas — {MODELO_ELEGIDO}\nTrain → Val → Backtest",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    return _save(fig, "02_degradacion_temporal_lr.png")


# -----------------------------------------------------------------------------
# 03 — Curvas ROC: Train / Val / Backtest superpuestas
# -----------------------------------------------------------------------------

def plot_roc_curvas(datos, probas):
    fig, ax = plt.subplots(figsize=(8, 7))

    for periodo in PERIODOS_ORDEN:
        _, y = datos[periodo]
        auc = roc_auc_score(y, probas[periodo])
        RocCurveDisplay.from_predictions(
            y, probas[periodo],
            name=f"{periodo}  (AUC = {auc:.4f})",
            ax=ax, color=COLORES_PERIODO[periodo],
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random  (AUC = 0.5000)")
    ax.set_title(
        f"Curvas ROC — {MODELO_ELEGIDO}\nTrain / Val / Backtest",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=11)
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "03_roc_lr.png")


# -----------------------------------------------------------------------------
# 04 — Curvas Precision-Recall: Train / Val / Backtest superpuestas
# -----------------------------------------------------------------------------

def plot_pr_curvas(datos, probas):
    fig, ax = plt.subplots(figsize=(8, 7))

    for periodo in PERIODOS_ORDEN:
        _, y = datos[periodo]
        ap = average_precision_score(y, probas[periodo])
        PrecisionRecallDisplay.from_predictions(
            y, probas[periodo],
            name=f"{periodo}  (PR-AUC = {ap:.4f})",
            ax=ax, color=COLORES_PERIODO[periodo],
        )

    _, y_bt = datos["Backtest"]
    prev = float(y_bt.mean())
    ax.axhline(prev, color="gray", ls="--", lw=1.2,
               label=f"Prevalencia Backtest = {prev:.2f}")
    ax.set_title(
        f"Curvas Precision-Recall — {MODELO_ELEGIDO}\nTrain / Val / Backtest",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _save(fig, "04_pr_lr.png")


# -----------------------------------------------------------------------------
# 05 — Matrices de confusión: Val y Backtest
# -----------------------------------------------------------------------------

def plot_confusion_matrices(datos, probas):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cmaps = {"Val": "Oranges", "Backtest": "Blues"}

    for ax, periodo in zip(axes, ["Val", "Backtest"]):
        _, y = datos[periodo]
        pred = (probas[periodo] >= 0.5).astype(int)
        cm   = confusion_matrix(y, pred)
        ConfusionMatrixDisplay(cm, display_labels=["Insatisfecho", "Satisfecho"]).plot(
            ax=ax, colorbar=False, cmap=cmaps[periodo],
        )
        p0 = precision_score(y, pred, pos_label=0, zero_division=0)
        r0 = recall_score(y, pred,    pos_label=0, zero_division=0)
        p1 = precision_score(y, pred, pos_label=1, zero_division=0)
        r1 = recall_score(y, pred,    pos_label=1, zero_division=0)
        ax.set_title(
            f"Confusión — {periodo}\n"
            f"Cls0  Prec={p0:.4f}  Rec={r0:.4f}\n"
            f"Cls1  Prec={p1:.4f}  Rec={r1:.4f}",
            fontsize=9, fontweight="bold",
        )

    plt.suptitle(
        f"Matrices de Confusión — {MODELO_ELEGIDO}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, "05_confusion_lr.png")


# -----------------------------------------------------------------------------
# 06 — Distribución de scores por periodo
# -----------------------------------------------------------------------------

def plot_distribucion_scores(datos, probas):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, periodo in zip(axes, PERIODOS_ORDEN):
        _, y = datos[periodo]
        p = probas[periodo]
        for clase, color, label in [
            (0, "#e74c3c", "Insatisfecho (cls 0)"),
            (1, "#2ecc71", "Satisfecho (cls 1)"),
        ]:
            ax.hist(p[y == clase], bins=40, alpha=0.6,
                    color=color, label=label, density=True)
        ax.axvline(0.5, color="black", ls="--", lw=1.5, label="Umbral 0.5")
        ax.set_xlabel("P(satisfecho)", fontsize=10)
        ax.set_title(periodo, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Densidad")
    plt.suptitle(
        f"Distribución de Scores — {MODELO_ELEGIDO}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, "06_distribucion_scores_lr.png")


# -----------------------------------------------------------------------------
# 07 — PSI y KS: estabilidad Val / Backtest vs Train
# -----------------------------------------------------------------------------

def plot_psi_ks(probas):
    from scipy.stats import ks_2samp

    periodos_cmp = ["Val", "Backtest"]
    ref = probas["Train"]

    psi_vals = []
    ks_stats = []
    for p in periodos_cmp:
        act = probas[p]
        psi_vals.append(_psi(ref, act))
        stat, _ = ks_2samp(ref, act)
        ks_stats.append(stat)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    bars = ax1.bar(
        periodos_cmp, psi_vals,
        color=[COLORES_PERIODO[p] for p in periodos_cmp],
        width=0.4, alpha=0.85,
    )
    ax1.bar_label(bars, fmt="%.4f", fontsize=11, fontweight="bold", padding=4)
    ax1.axhline(0.10, color="orange", ls="--", lw=1.5, label="PSI = 0.10 (alerta)")
    ax1.axhline(0.25, color="red",    ls="--", lw=1.5, label="PSI = 0.25 (crítico)")
    ax1.set_ylim(0, max(psi_vals + [0.05]) * 1.5 + 0.02)
    ax1.set_ylabel("PSI (vs Train)", fontsize=10)
    ax1.set_title("Population Stability Index\n(vs Train como referencia)",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    bars = ax2.bar(
        periodos_cmp, ks_stats,
        color=[COLORES_PERIODO[p] for p in periodos_cmp],
        width=0.4, alpha=0.85,
    )
    ax2.bar_label(bars, fmt="%.4f", fontsize=11, fontweight="bold", padding=4)
    ax2.set_ylim(0, max(ks_stats + [0.02]) * 1.5 + 0.01)
    ax2.set_ylabel("KS Statistic", fontsize=10)
    ax2.set_title("Kolmogorov-Smirnov Test\n(estabilidad de scores vs Train)",
                  fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Estabilidad de Distribución — {MODELO_ELEGIDO}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, "07_psi_ks_lr.png"), psi_vals, ks_stats


# -----------------------------------------------------------------------------
# 08 — Diagrama de calibración (Backtest)
# -----------------------------------------------------------------------------

def plot_calibracion(datos, probas):
    _, y_bt = datos["Backtest"]
    p = probas["Backtest"]

    frac_pos, mean_pred = calibration_curve(y_bt, p, n_bins=10, strategy="uniform")
    bs  = brier_score_loss(y_bt, p)
    auc = roc_auc_score(y_bt, p)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Calibración perfecta")
    ax.plot(mean_pred, frac_pos, "o-",
            color=COLOR_LR, lw=2.5, markersize=8,
            label=f"{MODELO_ELEGIDO}  (Brier = {bs:.4f})")
    ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.10, color=COLOR_LR)
    ax.set_xlabel("Probabilidad media predicha", fontsize=11)
    ax.set_ylabel("Fracción de positivos", fontsize=11)
    ax.set_title(
        f"Diagrama de Calibración — Backtest\n"
        f"{MODELO_ELEGIDO}   AUC = {auc:.4f}   Brier = {bs:.4f}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return _save(fig, "08_calibracion_lr.png")


# -----------------------------------------------------------------------------
# 09 — Análisis de umbral óptimo (Backtest)
# -----------------------------------------------------------------------------

def plot_umbral_optimo(datos, probas):
    _, y_bt = datos["Backtest"]
    p_bt = probas["Backtest"]

    t0, f1_0_opt, umbrales, f1s_0 = _umbral_optimo_f1(y_bt, p_bt, pos_label=0)
    t1, f1_1_opt, _,        f1s_1 = _umbral_optimo_f1(y_bt, p_bt, pos_label=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (f1s, t_opt, f1_opt, cls, color_opt) in zip(axes, [
        (f1s_0, t0, f1_0_opt, "Clase 0 — Insatisfecho", "#e74c3c"),
        (f1s_1, t1, f1_1_opt, "Clase 1 — Satisfecho",   "#27ae60"),
    ]):
        ax.plot(umbrales, f1s, color=COLOR_LR, lw=2.5, label=f"F1  {cls}")
        ax.fill_between(umbrales, f1s, alpha=0.12, color=COLOR_LR)
        ax.axvline(t_opt, color=color_opt, ls="--", lw=2,
                   label=f"Umbral óptimo = {t_opt:.2f}")
        ax.axvline(0.5,   color="gray",    ls=":",  lw=1.5,
                   label="Umbral estándar 0.5")
        ax.annotate(
            f"t* = {t_opt:.2f}\nF1 = {f1_opt:.4f}",
            (t_opt, f1_opt),
            textcoords="offset points", xytext=(12, -20),
            fontsize=10, color=color_opt, fontweight="bold",
        )
        ax.set_xlabel("Umbral de clasificación", fontsize=11)
        ax.set_ylabel(f"F1  {cls}", fontsize=10)
        ax.set_title(f"Umbral Óptimo — {cls}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(
        f"Análisis de Umbral Óptimo — Backtest\n{MODELO_ELEGIDO}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, "09_umbral_optimo_lr.png")


# -----------------------------------------------------------------------------
# 10 — Heatmap completo de métricas: Train / Val / Backtest
# -----------------------------------------------------------------------------

def plot_heatmap_metricas(resultados):
    METRICAS_HM = [
        "auc", "avg_prec",
        "prec_0", "recall_0",
        "prec_1", "recall_1",
        "f1_0", "f1_1", "f1_macro", "accuracy", "brier",
    ]
    LABELS_HM = [
        "AUC-ROC", "PR-AUC",
        "Precision Cls0", "Recall Cls0",
        "Precision Cls1", "Recall Cls1",
        "F1 Cls0", "F1 Cls1", "F1 Macro", "Accuracy", "Brier↓",
    ]

    data = np.array([
        [resultados[p][m] for p in PERIODOS_ORDEN]
        for m in METRICAS_HM
    ])
    # Invertir Brier para escala coherente (mayor = mejor)
    data[-1] = 1 - data[-1]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(PERIODOS_ORDEN)))
    ax.set_xticklabels(PERIODOS_ORDEN, fontsize=12)
    ax.set_yticks(range(len(LABELS_HM)))
    ax.set_yticklabels(LABELS_HM, fontsize=10)
    ax.set_title(
        f"Heatmap de Métricas — {MODELO_ELEGIDO}\n"
        "Train / Val / Backtest   (Brier mostrado como 1-Brier)",
        fontsize=11, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, fraction=0.04)

    for i in range(len(METRICAS_HM)):
        for j in range(len(PERIODOS_ORDEN)):
            val = resultados[PERIODOS_ORDEN[j]][METRICAS_HM[i]]
            txt_color = "black" if 0.35 < data[i, j] < 0.80 else "white"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=9, color=txt_color)

    plt.tight_layout()
    return _save(fig, "10_heatmap_metricas_lr.png")


# -----------------------------------------------------------------------------
# Exportar predicciones del Backtest a CSV / Excel
# -----------------------------------------------------------------------------

def guardar_predicciones_backtest(datos, probas, umbral_cls0: float = 0.5):
    X_bt, y_bt = datos["Backtest"]

    proba  = probas["Backtest"]
    pred05 = (proba >= 0.5).astype(int)
    pred_t = (proba >= umbral_cls0).astype(int)

    etiqueta = {0: "Insatisfecho", 1: "Satisfecho"}

    df = X_bt.copy().reset_index(drop=True)
    df["y_real"]               = y_bt.values
    df["y_real_etiqueta"]      = y_bt.map(etiqueta).values
    df["proba_satisfecho"]     = proba.round(6)
    df["prediccion"]           = pred05
    df["prediccion_etiqueta"]  = pd.Series(pred05).map(etiqueta).values
    df["correcto"]             = (pred05 == y_bt.values).astype(int)
    df["prediccion_umbral_opt"] = pred_t
    df["prediccion_umbral_opt_etiqueta"] = pd.Series(pred_t).map(etiqueta).values
    df["correcto_umbral_opt"]  = (pred_t == y_bt.values).astype(int)

    # Ordenar: primero los errores (incorrecto al inicio para revisión fácil)
    df = df.sort_values("correcto").reset_index(drop=True)

    path_csv  = REPORTES / "predicciones_backtest.csv"
    path_xlsx = REPORTES / "predicciones_backtest.xlsx"

    df.to_csv(path_csv, index=False, encoding="utf-8-sig")
    print(f"   -> data/reportes/predicciones_backtest.csv  ({len(df)} filas)")

    try:
        df.to_excel(path_xlsx, index=False, sheet_name="Backtest_Predicciones")
        print(f"   -> data/reportes/predicciones_backtest.xlsx  ({len(df)} filas)")
    except ImportError:
        print("   [!] openpyxl no instalado — solo se generó el CSV (pip install openpyxl)")

    # Mini resumen por clase real
    resumen = df.groupby("y_real_etiqueta").agg(
        total=("y_real", "count"),
        correctos=("correcto", "sum"),
    )
    resumen["tasa_acierto"] = (resumen["correctos"] / resumen["total"]).round(4)
    print("\n   Resumen por clase real (umbral 0.5):")
    print(resumen.to_string())

    return df, str(path_csv), str(path_xlsx)


# -----------------------------------------------------------------------------
# Actualizar y_backtest con probabilidades del modelo
# -----------------------------------------------------------------------------

def actualizar_y_backtest_con_proba(proba: np.ndarray):
    ruta_csv  = DATA / "y_backtest.csv"
    ruta_xlsx = DATA / "y_backtest.xlsx"

    y_df = pd.read_csv(ruta_csv)
    y_df["proba_satisfecho"] = proba.round(6)

    y_df.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"   -> data/master/y_backtest.csv  (proba_satisfecho añadida, {len(y_df)} filas)")

    try:
        y_df.to_excel(ruta_xlsx, index=False, engine="openpyxl")
        print(f"   -> data/master/y_backtest.xlsx  (proba_satisfecho añadida)")
    except Exception as e:
        print(f"   [!] No se pudo actualizar y_backtest.xlsx: {e}")


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------

def main():
    t_inicio = time.time()
    print("=" * 60)
    print("08_backtest.py — Backtesting Sprint 4")
    print(f"Modelo seleccionado: {MODELO_ELEGIDO}")
    print("=" * 60)

    print("\n[1] Cargando datos (Train / Val / Backtest)...")
    datos = cargar_datos()

    print("\n[2] Cargando modelo Optuna (cls0)...")
    modelo = cargar_modelo()

    print("\n[3] Calculando métricas...")
    resultados, probas = calcular_metricas_lr(modelo, datos)

    print("\n[3b] Actualizando y_backtest con probabilidades del modelo...")
    actualizar_y_backtest_con_proba(probas["Backtest"])

    # Resumen en consola — ambas clases
    print("\n" + "-" * 84)
    print(f"{'Periodo':<12} {'AUC':>6} {'PR-AUC':>7} "
          f"{'Prec0':>7} {'Rec0':>7} {'F1_0':>6} "
          f"{'Prec1':>7} {'Rec1':>7} {'F1_1':>6} {'F1Mac':>7}")
    print("-" * 84)
    for periodo in PERIODOS_ORDEN:
        m = resultados[periodo]
        print(f"  {periodo:<10} {m['auc']:>6.4f} {m['avg_prec']:>7.4f} "
              f"{m['prec_0']:>7.4f} {m['recall_0']:>7.4f} {m['f1_0']:>6.4f} "
              f"{m['prec_1']:>7.4f} {m['recall_1']:>7.4f} {m['f1_1']:>6.4f} "
              f"{m['f1_macro']:>7.4f}")

    # Gráficos
    print("\n[4] Generando gráficos...\n")
    generados = []

    generados.append(plot_metricas_clave(resultados))
    generados.append(plot_degradacion_temporal(resultados))
    generados.append(plot_roc_curvas(datos, probas))
    generados.append(plot_pr_curvas(datos, probas))
    generados.append(plot_confusion_matrices(datos, probas))
    generados.append(plot_distribucion_scores(datos, probas))

    try:
        nombre_g, psi_vals, ks_stats = plot_psi_ks(probas)
        generados.append(nombre_g)
    except ImportError:
        print("   [!] scipy no disponible — omitiendo PSI/KS (pip install scipy)")
        psi_vals = ks_stats = {}

    generados.append(plot_calibracion(datos, probas))
    generados.append(plot_umbral_optimo(datos, probas))
    generados.append(plot_heatmap_metricas(resultados))

    # Umbral óptimo en Backtest — ambas clases
    _, y_bt = datos["Backtest"]
    t0, f1_0_opt, _, _ = _umbral_optimo_f1(y_bt, probas["Backtest"], pos_label=0)
    t1, f1_1_opt, _, _ = _umbral_optimo_f1(y_bt, probas["Backtest"], pos_label=1)
    umbral_opt = {
        "cls0": {"umbral": round(float(t0), 3), "f1_0": round(float(f1_0_opt), 4)},
        "cls1": {"umbral": round(float(t1), 3), "f1_1": round(float(f1_1_opt), 4)},
    }

    # Persistencia
    duracion = round(time.time() - t_inicio, 2)

    res_serial = {
        MODELO_ELEGIDO: {
            periodo: {k: float(v) for k, v in metricas.items()}
            for periodo, metricas in resultados.items()
        }
    }
    with open(REPORTES / "resultados_backtest.json", "w") as f:
        json.dump(res_serial, f, indent=2)

    estado = {
        "script":             "08_backtest",
        "timestamp":          datetime.now().isoformat(),
        "duracion_seg":       duracion,
        "modelo_elegido":     MODELO_ELEGIDO,
        "periodos_evaluados": PERIODOS_ORDEN,
        "metricas_backtest": {
            "auc":      resultados["Backtest"]["auc"],
            "avg_prec": resultados["Backtest"]["avg_prec"],
            "prec_0":   resultados["Backtest"]["prec_0"],
            "recall_0": resultados["Backtest"]["recall_0"],
            "f1_0":     resultados["Backtest"]["f1_0"],
            "prec_1":   resultados["Backtest"]["prec_1"],
            "recall_1": resultados["Backtest"]["recall_1"],
            "f1_1":     resultados["Backtest"]["f1_1"],
            "f1_macro": resultados["Backtest"]["f1_macro"],
        },
        "umbral_optimo_backtest": umbral_opt,
        "graficos_generados":  generados,
        "directorio_salida":   "reports/sprint4",
    }
    with open(REPORTES / "estado_backtest.json", "w") as f:
        json.dump(estado, f, indent=2)

    # Exportar predicciones
    print("\n[5] Exportando predicciones del Backtest...\n")
    _, path_csv, path_xlsx = guardar_predicciones_backtest(
        datos, probas, umbral_cls0=umbral_opt["cls0"]["umbral"]
    )

    # Agregar rutas al estado
    estado["archivos_predicciones"] = {
        "csv":  "data/reportes/predicciones_backtest.csv",
        "xlsx": "data/reportes/predicciones_backtest.xlsx",
    }
    with open(REPORTES / "estado_backtest.json", "w") as f:
        json.dump(estado, f, indent=2)

    print(f"\n{'-'*60}")
    print(f"[OK] {MODELO_ELEGIDO} — Backtest")
    print(f"   AUC-ROC          = {resultados['Backtest']['auc']:.4f}")
    print(f"   PR-AUC           = {resultados['Backtest']['avg_prec']:.4f}")
    print(f"   Precision  Cls0  = {resultados['Backtest']['prec_0']:.4f}")
    print(f"   Recall     Cls0  = {resultados['Backtest']['recall_0']:.4f}")
    print(f"   F1         Cls0  = {resultados['Backtest']['f1_0']:.4f}")
    print(f"   Precision  Cls1  = {resultados['Backtest']['prec_1']:.4f}")
    print(f"   Recall     Cls1  = {resultados['Backtest']['recall_1']:.4f}")
    print(f"   F1         Cls1  = {resultados['Backtest']['f1_1']:.4f}")
    print(f"   F1 Macro         = {resultados['Backtest']['f1_macro']:.4f}")
    print(f"   Umbral opt Cls0  = {umbral_opt['cls0']['umbral']}  (F1={umbral_opt['cls0']['f1_0']:.4f})")
    print(f"   Umbral opt Cls1  = {umbral_opt['cls1']['umbral']}  (F1={umbral_opt['cls1']['f1_1']:.4f})")
    print(f"\n08_backtest.py completado en {duracion}s  ({len(generados)} gráficos)")
    print(f"   -> data/reportes/resultados_backtest.json")
    print(f"   -> data/reportes/estado_backtest.json")
    print(f"   -> data/reportes/predicciones_backtest.csv")
    print(f"   -> data/reportes/predicciones_backtest.xlsx")
    print(f"   -> reports/sprint4/  ({len(generados)} PNG)")


if __name__ == "__main__":
    main()
