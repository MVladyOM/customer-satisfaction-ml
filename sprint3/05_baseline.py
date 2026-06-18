from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

ROOT = Path(__file__).resolve().parent.parent
REPORTES = ROOT / "data" / "reportes"
REPORTES.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data" / "master"
SEED            = 2357
UMBRAL_OVERFIT  = 0.05   # diferencia train-val que dispara la advertencia


def cargar_datos():
    X_train = pd.read_csv(DATA / "X_train.csv")
    X_val   = pd.read_csv(DATA / "X_val.csv")
    X_bt    = pd.read_csv(DATA / "X_backtest.csv")
    y_train = pd.read_csv(DATA / "y_train.csv").squeeze()
    y_val   = pd.read_csv(DATA / "y_val.csv").squeeze()
    y_bt    = pd.read_csv(DATA / "y_backtest.csv").squeeze()
    print(f"Train:     {X_train.shape}  | positivos: {y_train.mean():.2%}")
    print(f"Val:       {X_val.shape}    | positivos: {y_val.mean():.2%}")
    print(f"Backtest:  {X_bt.shape}     | positivos: {y_bt.mean():.2%}")
    return X_train, X_val, X_bt, y_train, y_val, y_bt


def definir_modelos():
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-2),
        "linearSVM": CalibratedClassifierCV(
            LinearSVC(random_state=SEED), method="sigmoid"
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, random_state=SEED, n_jobs=-2
        ),
        "SGD": SGDClassifier(loss="log_loss", random_state=SEED),
        "HGBC": HistGradientBoostingClassifier(random_state=SEED),
    }
    if CatBoostClassifier is not None:
        modelos["CatBoost"] = CatBoostClassifier(
            random_state=SEED, verbose=False, task_type="CPU"
        )
    if LGBMClassifier is not None:
        modelos["LightGBM"] = LGBMClassifier(
            random_state=SEED, verbosity=-1, n_jobs=-2
        )
    return modelos


def _metricas_set(y_true, pred, proba):
    """Calcula el bloque completo de métricas sobre un conjunto."""
    rec0 = recall_score(y_true, pred, pos_label=0, zero_division=0)
    rec1 = recall_score(y_true, pred, pos_label=1, zero_division=0)
    return {
        "AUC-ROC":  round(roc_auc_score(y_true, proba), 4),
        "AvgPrec":  round(average_precision_score(y_true, proba), 4),
        "Brier":    round(brier_score_loss(y_true, proba), 4),
        "F1_macro": round(f1_score(y_true, pred, average="macro", zero_division=0), 4),
        # Desbalanceo
        "MCC":      round(matthews_corrcoef(y_true, pred), 4),
        "Kappa":    round(cohen_kappa_score(y_true, pred), 4),
        "BalAcc":   round(balanced_accuracy_score(y_true, pred), 4),
        "Gmean":    round(float(np.sqrt(rec0 * rec1)), 4),
        # Clase 1 — Satisfecho
        "F1_1":     round(f1_score(y_true, pred,        pos_label=1, zero_division=0), 4),
        "Prec_1":   round(precision_score(y_true, pred, pos_label=1, zero_division=0), 4),
        "Recall_1": round(rec1, 4),
        # Clase 0 — Insatisfecho
        "F1_0":     round(f1_score(y_true, pred,        pos_label=0, zero_division=0), 4),
        "Prec_0":   round(precision_score(y_true, pred, pos_label=0, zero_division=0), 4),
        "Recall_0": round(rec0, 4),
    }


def evaluar_modelo(nombre, modelo, X_tr, y_tr, X_eval, y_eval):
    modelo.fit(X_tr, y_tr)

    # ── Evaluación en validación ────────────────────────────────────────────
    proba_val = modelo.predict_proba(X_eval)[:, 1]
    pred_val  = (proba_val >= 0.5).astype(int)
    m_val     = _metricas_set(y_eval, pred_val, proba_val)

    # ── Evaluación en train (para detección de overfitting) ─────────────────
    proba_tr = modelo.predict_proba(X_tr)[:, 1]
    pred_tr  = (proba_tr >= 0.5).astype(int)
    m_tr     = _metricas_set(y_tr, pred_tr, proba_tr)

    # ── Deltas y flag ────────────────────────────────────────────────────────
    delta_auc  = round(m_tr["AUC-ROC"] - m_val["AUC-ROC"], 4)
    delta_f1   = round(m_tr["F1_macro"] - m_val["F1_macro"], 4)
    delta_mcc  = round(m_tr["MCC"]      - m_val["MCC"], 4)
    overfit    = bool(delta_auc > UMBRAL_OVERFIT or delta_f1 > UMBRAL_OVERFIT)

    result = {"modelo": nombre, "overfit": overfit}
    for k, v in m_val.items():
        result[k] = v
    for k, v in m_tr.items():
        result[f"{k}_tr"] = v
    result["delta_auc"] = delta_auc
    result["delta_f1"]  = delta_f1
    result["delta_mcc"] = delta_mcc
    result["_proba"]    = proba_val
    result["_modelo"]   = modelo
    return result


def analisis_umbral(resultados, y_eval):
    umbrales = []
    thresholds = np.linspace(0.01, 0.99, 200)
    for r in resultados:
        proba = r["_proba"]
        f1s = []
        for t in thresholds:
            pred_t = (proba >= t).astype(int)
            if pred_t.sum() == 0 or pred_t.sum() == len(pred_t):
                f1s.append(0)
                continue
            f1s.append(f1_score(y_eval, pred_t, zero_division=0))
        idx = int(np.argmax(f1s))
        t_opt = float(thresholds[idx])
        pred_opt = (proba >= t_opt).astype(int)
        umbrales.append({
            "modelo":        r["modelo"],
            "umbral_optimo": round(t_opt, 3),
            "f1_optimo":     round(float(f1s[idx]), 4),
            "delta_f1":      round(float(f1s[idx]) - r["F1_1"], 4),
            "precision_opt": round(float(precision_score(y_eval, pred_opt, zero_division=0)), 4),
            "recall_opt":    round(float(recall_score(y_eval, pred_opt, zero_division=0)), 4),
        })
    return umbrales


def main():
    t_inicio = time.time()
    print("=" * 55)
    print("05_baseline.py — Baseline multimodelo Sprint 3")
    print("=" * 55)

    X_train, X_val, X_bt, y_train, y_val, y_bt = cargar_datos()
    modelos = definir_modelos()
    print(f"\n{len(modelos)} modelos definidos\n")

    resultados_val = []
    for nombre, modelo in modelos.items():
        t0 = time.time()
        print(f"  Entrenando {nombre}...", end=" ", flush=True)
        r = evaluar_modelo(nombre, modelo, X_train, y_train, X_val, y_val)
        resultados_val.append(r)
        print(f"AUC={r['AUC-ROC']}  F1_0={r['F1_0']}  F1_1={r['F1_1']}  ({time.time()-t0:.1f}s)")

    print("\nAnálisis de umbral en validación...")
    umbrales_val = analisis_umbral(resultados_val, y_val)

    mejor = max(resultados_val, key=lambda x: x["AUC-ROC"])
    print(f"\nMejor modelo (AUC-ROC): {mejor['modelo']}  AUC={mejor['AUC-ROC']}")

    df_all = pd.DataFrame(resultados_val).sort_values("AUC-ROC", ascending=False)

    # ── Tabla desbalanceo ────────────────────────────────────────────────────
    print("\n--- Metricas de desbalanceo (validacion) ---")
    cols_imb = ["modelo", "AUC-ROC", "AvgPrec", "MCC", "Kappa", "BalAcc", "Gmean", "Brier"]
    print(df_all[cols_imb].to_string(index=False))

    # ── Tabla clase 1 ────────────────────────────────────────────────────────
    print("\n--- Clase 1 (Satisfecho) ---")
    cols1 = ["modelo", "AUC-ROC", "F1_1", "Prec_1", "Recall_1", "F1_macro"]
    print(df_all[cols1].to_string(index=False))

    # ── Tabla clase 0 ────────────────────────────────────────────────────────
    print("\n--- Clase 0 (Insatisfecho) ---")
    cols0 = ["modelo", "AUC-ROC", "F1_0", "Prec_0", "Recall_0"]
    print(df_all[cols0].to_string(index=False))

    # ── Tabla overfitting ────────────────────────────────────────────────────
    print(f"\n--- Overfitting (umbral={UMBRAL_OVERFIT}) ---")
    cols_ov = ["modelo", "AUC-ROC", "AUC-ROC_tr", "delta_auc",
               "F1_macro", "F1_macro_tr", "delta_f1",
               "MCC", "MCC_tr", "delta_mcc", "overfit"]
    df_ov = df_all[cols_ov].copy()
    df_ov["overfit"] = df_ov["overfit"].map({True: "*** SI ***", False: "no"})
    print(df_ov.to_string(index=False))

    registros = []
    for r in resultados_val:
        umb = next(u for u in umbrales_val if u["modelo"] == r["modelo"])
        registros.append({
            "modelo":          r["modelo"],
            # Validación
            "auc_val":         r["AUC-ROC"],
            "gini_val":        round(2 * r["AUC-ROC"] - 1, 4),
            "avg_prec_val":    r["AvgPrec"],
            "f1_macro_val":    r["F1_macro"],
            "brier_val":       r["Brier"],
            "mcc_val":         r["MCC"],
            "kappa_val":       r["Kappa"],
            "bal_acc_val":     r["BalAcc"],
            "gmean_val":       r["Gmean"],
            # Clase 1 val
            "f1_1_val":        r["F1_1"],
            "prec_1_val":      r["Prec_1"],
            "recall_1_val":    r["Recall_1"],
            # Clase 0 val
            "f1_0_val":        r["F1_0"],
            "prec_0_val":      r["Prec_0"],
            "recall_0_val":    r["Recall_0"],
            # Train (overfitting)
            "auc_train":       r["AUC-ROC_tr"],
            "f1_macro_train":  r["F1_macro_tr"],
            "mcc_train":       r["MCC_tr"],
            "delta_auc":       r["delta_auc"],
            "delta_f1":        r["delta_f1"],
            "delta_mcc":       r["delta_mcc"],
            "overfit":         r["overfit"],
            # Umbral
            "umbral_optimo":   umb["umbral_optimo"],
            "f1_umbral_opt":   umb["f1_optimo"],
        })

    with open(REPORTES / "resultados_baseline.json", "w") as f:
        json.dump(registros, f, indent=2)

    modelos_dict = {r["modelo"]: r["_modelo"] for r in resultados_val}
    joblib.dump(modelos_dict, REPORTES / "modelos_baseline.pkl")

    n_overfit = sum(1 for r in resultados_val if r["overfit"])
    duracion = round(time.time() - t_inicio, 2)
    estado = {
        "script":           "05_baseline",
        "timestamp":        datetime.now().isoformat(),
        "duracion_seg":     duracion,
        "seed":             SEED,
        "umbral_overfit":   UMBRAL_OVERFIT,
        "n_modelos":        len(modelos),
        "n_overfit":        n_overfit,
        "mejor_modelo":     mejor["modelo"],
        "auc_mejor":        mejor["AUC-ROC"],
        "mcc_mejor":        mejor["MCC"],
        "bal_acc_mejor":    mejor["BalAcc"],
        "gmean_mejor":      mejor["Gmean"],
        "f1_0_mejor":       mejor["F1_0"],
        "f1_1_mejor":       mejor["F1_1"],
        "f1_macro_mejor":   mejor["F1_macro"],
        "overfit_mejor":    mejor["overfit"],
        "delta_auc_mejor":  mejor["delta_auc"],
    }
    with open(REPORTES / "estado_baseline.json", "w") as f:
        json.dump(estado, f, indent=2)

    print(f"\n05_baseline.py completado en {duracion}s")
    print(f"   -> data/reportes/resultados_baseline.json")
    print(f"   -> data/reportes/modelos_baseline.pkl")
    print(f"   -> data/reportes/estado_baseline.json")


if __name__ == "__main__":
    main()