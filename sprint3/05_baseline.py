from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

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
    brier_score_loss,
    confusion_matrix,
    f1_score,
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
SEED = 2357


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


def evaluar_modelo(nombre, modelo, X_tr, y_tr, X_eval, y_eval):
    modelo.fit(X_tr, y_tr)
    proba = modelo.predict_proba(X_eval)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return {
        "modelo":   nombre,
        "AUC-ROC":  round(roc_auc_score(y_eval, proba), 4),
        "Gini":     round(2 * roc_auc_score(y_eval, proba) - 1, 4),
        "F1":       round(f1_score(y_eval, pred), 4),
        "Precision":round(precision_score(y_eval, pred), 4),
        "Recall":   round(recall_score(y_eval, pred), 4),
        "Brier":    round(brier_score_loss(y_eval, proba), 4),
        "_proba":   proba,
        "_modelo":  modelo,
    }


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
            "delta_f1":      round(float(f1s[idx]) - r["F1"], 4),
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
        print(f"AUC={r['AUC-ROC']}  F1={r['F1']}  ({time.time()-t0:.1f}s)")

    print("\nAnálisis de umbral en validación...")
    umbrales_val = analisis_umbral(resultados_val, y_val)

    mejor = max(resultados_val, key=lambda x: x["AUC-ROC"])
    print(f"\nMejor modelo (AUC-ROC): {mejor['modelo']}  AUC={mejor['AUC-ROC']}")

    cols = ["modelo", "AUC-ROC", "Gini", "F1", "Precision", "Recall", "Brier"]
    df = pd.DataFrame(resultados_val)[cols].sort_values("AUC-ROC", ascending=False)
    print("\n" + df.to_string(index=False))

    registros = []
    for r in resultados_val:
        umb = next(u for u in umbrales_val if u["modelo"] == r["modelo"])
        registros.append({
            "modelo":          r["modelo"],
            "auc_val":         r["AUC-ROC"],
            "gini_val":        r["Gini"],
            "f1_val":          r["F1"],
            "precision_val":   r["Precision"],
            "recall_val":      r["Recall"],
            "brier_val":       r["Brier"],
            "umbral_optimo":   umb["umbral_optimo"],
            "f1_umbral_opt":   umb["f1_optimo"],
            "delta_f1":        umb["delta_f1"],
        })

    with open(REPORTES / "resultados_baseline.json", "w") as f:
        json.dump(registros, f, indent=2)

    duracion = round(time.time() - t_inicio, 2)
    estado = {
        "script":          "05_baseline",
        "timestamp":       datetime.now().isoformat(),
        "duracion_seg":    duracion,
        "seed":            SEED,
        "n_modelos":       len(modelos),
        "mejor_modelo":    mejor["modelo"],
        "auc_mejor":       mejor["AUC-ROC"],
        "gini_mejor":      mejor["Gini"],
        "f1_mejor":        mejor["F1"],
    }
    with open(REPORTES / "estado_baseline.json", "w") as f:
        json.dump(estado, f, indent=2)

    print(f"\n✅ 05_baseline.py completado en {duracion}s")
    print(f"   → data/reportes/resultados_baseline.json")
    print(f"   → data/reportes/estado_baseline.json")


if __name__ == "__main__":
    main()