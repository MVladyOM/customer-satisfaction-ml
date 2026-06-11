"""
=============================================================================
Script 05 - Hiperparametrización con Optuna
Proyecto : Predicción de Satisfacción del Cliente — Olist
Grupo 1  | Sprint 2
=============================================================================
Responsabilidad:
  - Optimizar 5 modelos baseline usando Optuna (TPE, 100 trials c/u)
  - Métrica objetivo: F1 de clase 0 (insatisfechos)
  - Re-entrenar con mejores parámetros y evaluar en val y backtest
  - Guardar modelos entrenados, estudios Optuna y tabla de resultados

Entrada : data/master/X_train.csv / y_train.csv
          data/master/X_val.csv   / y_val.csv
          data/master/X_backtest.csv / y_backtest.csv

Salida  : data/reportes/resultados_optuna.json
          data/reportes/mejores_params.json
          data/reportes/modelos_optuna.pkl
          data/reportes/estudios_optuna.pkl
          data/reportes/estado_optuna.json
=============================================================================
"""

import os
import json
import time
import pickle
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
MASTER_PATH   = os.path.join("data", "master")
REPORTES_PATH = os.path.join("data", "reportes")

# ---------------------------------------------------------------------------
# Parámetros globales
# ---------------------------------------------------------------------------
SEED      = 2357
N_TRIALS  = 100
POS_LABEL = 0      # clase de interés: insatisfecho (0)


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------

def cargar_datos() -> tuple:
    archivos = {
        "X_train":    os.path.join(MASTER_PATH, "X_train.csv"),
        "y_train":    os.path.join(MASTER_PATH, "y_train.csv"),
        "X_val":      os.path.join(MASTER_PATH, "X_val.csv"),
        "y_val":      os.path.join(MASTER_PATH, "y_val.csv"),
        "X_backtest": os.path.join(MASTER_PATH, "X_backtest.csv"),
        "y_backtest": os.path.join(MASTER_PATH, "y_backtest.csv"),
    }
    for nombre, ruta in archivos.items():
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró {ruta}. Ejecuta primero Script 04.")

    X_train    = pd.read_csv(archivos["X_train"])
    y_train    = pd.read_csv(archivos["y_train"]).squeeze()
    X_val      = pd.read_csv(archivos["X_val"])
    y_val      = pd.read_csv(archivos["y_val"]).squeeze()
    X_backtest = pd.read_csv(archivos["X_backtest"])
    y_backtest = pd.read_csv(archivos["y_backtest"]).squeeze()

    print(f"  X_train    : {X_train.shape}  | insatisfechos: {(y_train==0).mean():.2%}")
    print(f"  X_val      : {X_val.shape}  | insatisfechos: {(y_val==0).mean():.2%}")
    print(f"  X_backtest : {X_backtest.shape}  | insatisfechos: {(y_backtest==0).mean():.2%}")

    # scale_pos_weight para XGBoost / LightGBM
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = round(pos / neg, 4)
    print(f"  scale_pos_weight: {spw}")

    return X_train, y_train, X_val, y_val, X_backtest, y_backtest, spw


# ---------------------------------------------------------------------------
# 2. MÉTRICAS
# ---------------------------------------------------------------------------

def calcular_metricas(y_true, y_pred, y_proba) -> dict:
    return {
        "f1_0"     : round(float(f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)), 4),
        "recall_0" : round(float(recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)), 4),
        "prec_0"   : round(float(precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)), 4),
        "f1_macro" : round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "auc"      : round(float(roc_auc_score(y_true, y_proba)), 4),
        "accuracy" : round(float(accuracy_score(y_true, y_pred)), 4),
    }


# ---------------------------------------------------------------------------
# 3. OBJECTIVE FUNCTIONS
# ---------------------------------------------------------------------------

def hacer_objectives(X_train, y_train, X_val, y_val, spw):
    """Cierra sobre los datos y devuelve dict nombre→función objective."""

    sw_array = np.where(
        y_train.to_numpy().ravel() == POS_LABEL,
        len(y_train) / (2 * (y_train == POS_LABEL).sum()),
        len(y_train) / (2 * (y_train != POS_LABEL).sum()),
    ).ravel()

    def objective_lr(trial):
        params = {
            "C"           : trial.suggest_float("C", 1e-3, 10, log=True),
            "solver"      : trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "max_iter"    : 1000,
            "random_state": SEED,
        }
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_rf(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"        : trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features"     : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight"     : trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state"     : SEED,
            "n_jobs"           : -1,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_gb(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"       : trial.suggest_int("max_depth", 2, 8),
            "learning_rate"   : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features"    : trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state"    : SEED,
        }
        use_sw = trial.suggest_categorical("use_sample_weight", [True, False])
        model  = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sw_array if use_sw else None)
        proba = model.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_xgb(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"       : trial.suggest_int("max_depth", 2, 10),
            "learning_rate"   : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1, spw, round(spw * 1.5, 4)]),
            "random_state"    : SEED,
            "eval_metric"     : "logloss",
            "verbosity"       : 0,
            "n_jobs"          : -1,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_lgbm(trial):
        params = {
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 500),
            "max_depth"       : trial.suggest_int("max_depth", 2, 10),
            "learning_rate"   : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves"      : trial.suggest_int("num_leaves", 20, 150),
            "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "class_weight"    : trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state"    : SEED,
            "verbosity"       : -1,
            "n_jobs"          : -1,
        }
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    return {
        "Logistic Regression": objective_lr,
        "Random Forest"      : objective_rf,
        "Gradient Boosting"  : objective_gb,
        "XGBoost"            : objective_xgb,
        "LightGBM"           : objective_lgbm,
    }


# ---------------------------------------------------------------------------
# 4. EJECUTAR ESTUDIOS OPTUNA
# ---------------------------------------------------------------------------

def ejecutar_estudios(objectives: dict) -> dict:
    estudios = {}
    for nombre, objective in objectives.items():
        print(f"\n  ▶ Optimizando {nombre} ({N_TRIALS} trials)...")
        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            study_name=nombre,
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
        elapsed = time.time() - t0
        estudios[nombre] = study
        print(f"  ✓ Mejor F1(cls 0): {study.best_value:.4f} | {elapsed:.0f}s")
        print(f"    Params: {study.best_params}")
    return estudios


# ---------------------------------------------------------------------------
# 5. RE-ENTRENAR Y EVALUAR CON MEJORES PARÁMETROS
# ---------------------------------------------------------------------------

def reconstruir_modelo(nombre: str, params: dict, spw: float, y_train: pd.Series):
    p = params.copy()
    sw = np.where(
        y_train.to_numpy().ravel() == POS_LABEL,
        len(y_train) / (2 * (y_train == POS_LABEL).sum()),
        len(y_train) / (2 * (y_train != POS_LABEL).sum()),
    ).ravel()

    if nombre == "Logistic Regression":
        return LogisticRegression(**p, max_iter=1000, random_state=SEED), None

    if nombre == "Random Forest":
        return RandomForestClassifier(**p, random_state=SEED, n_jobs=-1), None

    if nombre == "Gradient Boosting":
        use_sw = p.pop("use_sample_weight")
        return GradientBoostingClassifier(**p, random_state=SEED), (sw if use_sw else None)

    if nombre == "XGBoost":
        return XGBClassifier(**p, random_state=SEED, eval_metric="logloss",
                             verbosity=0, n_jobs=-1), None

    if nombre == "LightGBM":
        return LGBMClassifier(**p, random_state=SEED, verbosity=-1, n_jobs=-1), None


def reentrenar_y_evaluar(
    estudios: dict, spw: float,
    X_train, y_train, X_val, y_val, X_backtest, y_backtest
) -> tuple:
    resultados = []
    modelos    = {}

    for nombre, study in estudios.items():
        best_params = study.best_params.copy()
        model, sw   = reconstruir_modelo(nombre, best_params, spw, y_train)

        if sw is not None:
            model.fit(X_train, y_train, sample_weight=sw)
        else:
            model.fit(X_train, y_train)

        # Métricas en validación
        proba_val = model.predict_proba(X_val)[:, 1]
        pred_val  = (proba_val >= 0.5).astype(int)
        m_val     = calcular_metricas(y_val, pred_val, proba_val)

        # Métricas en backtest
        proba_bt = model.predict_proba(X_backtest)[:, 1]
        pred_bt  = (proba_bt >= 0.5).astype(int)
        m_bt     = calcular_metricas(y_backtest, pred_bt, proba_bt)

        modelos[nombre] = {
            "model"    : model,
            "proba_val": proba_val.tolist(),
            "proba_bt" : proba_bt.tolist(),
            "pred_val" : pred_val.tolist(),
            "pred_bt"  : pred_bt.tolist(),
        }

        resultados.append({
            "modelo"        : nombre,
            "f1_0_val"      : m_val["f1_0"],
            "recall_0_val"  : m_val["recall_0"],
            "prec_0_val"    : m_val["prec_0"],
            "f1_macro_val"  : m_val["f1_macro"],
            "auc_val"       : m_val["auc"],
            "accuracy_val"  : m_val["accuracy"],
            "f1_0_bt"       : m_bt["f1_0"],
            "recall_0_bt"   : m_bt["recall_0"],
            "prec_0_bt"     : m_bt["prec_0"],
            "f1_macro_bt"   : m_bt["f1_macro"],
            "auc_bt"        : m_bt["auc"],
            "accuracy_bt"   : m_bt["accuracy"],
            "best_params"   : study.best_params,
        })

        print(
            f"  {nombre:<25} | val  F1(0)={m_val['f1_0']:.4f}  AUC={m_val['auc']:.4f}"
            f"  | bt  F1(0)={m_bt['f1_0']:.4f}  AUC={m_bt['auc']:.4f}"
        )

    return resultados, modelos


# ---------------------------------------------------------------------------
# 6. GUARDAR
# ---------------------------------------------------------------------------

def guardar_resultados(resultados: list, estudios: dict, modelos: dict, estado: dict):
    os.makedirs(REPORTES_PATH, exist_ok=True)

    # resultados_optuna.json
    with open(os.path.join(REPORTES_PATH, "resultados_optuna.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False, default=str)

    # mejores_params.json
    mejores = {r["modelo"]: r["best_params"] for r in resultados}
    with open(os.path.join(REPORTES_PATH, "mejores_params.json"), "w", encoding="utf-8") as f:
        json.dump(mejores, f, indent=2, ensure_ascii=False, default=str)

    # modelos_optuna.pkl — solo los objetos model + arrays de predicción
    modelos_pkl = {
        nombre: {
            "model"    : v["model"],
            "proba_val": v["proba_val"],
            "proba_bt" : v["proba_bt"],
            "pred_val" : v["pred_val"],
            "pred_bt"  : v["pred_bt"],
        }
        for nombre, v in modelos.items()
    }
    with open(os.path.join(REPORTES_PATH, "modelos_optuna.pkl"), "wb") as f:
        pickle.dump(modelos_pkl, f)

    # estudios_optuna.pkl — objetos optuna.Study para análisis de convergencia
    with open(os.path.join(REPORTES_PATH, "estudios_optuna.pkl"), "wb") as f:
        pickle.dump(estudios, f)

    # estado_optuna.json
    with open(os.path.join(REPORTES_PATH, "estado_optuna.json"), "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, ensure_ascii=False, default=str)

    print(f"  resultados_optuna.json  ({len(resultados)} modelos)")
    print(f"  mejores_params.json")
    print(f"  modelos_optuna.pkl")
    print(f"  estudios_optuna.pkl")
    print(f"  estado_optuna.json")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    inicio = datetime.now()
    print("=" * 80)
    print("SCRIPT 05 — HIPERPARAMETRIZACIÓN CON OPTUNA")
    print(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\n[1] Cargando datos ...")
    X_train, y_train, X_val, y_val, X_backtest, y_backtest, spw = cargar_datos()

    print("\n[2] Definiendo objective functions ...")
    objectives = hacer_objectives(X_train, y_train, X_val, y_val, spw)

    print(f"\n[3] Ejecutando estudios Optuna ({N_TRIALS} trials por modelo) ...")
    print("=" * 80)
    estudios = ejecutar_estudios(objectives)
    print("=" * 80)

    print("\n[4] Re-entrenando con mejores parámetros y evaluando ...")
    resultados, modelos = reentrenar_y_evaluar(
        estudios, spw, X_train, y_train, X_val, y_val, X_backtest, y_backtest
    )

    # Modelo ganador por F1(cls 0) en val
    ganador = max(resultados, key=lambda r: r["f1_0_val"])

    print("\n" + "=" * 80)
    print("TABLA COMPARATIVA — VALIDACIÓN")
    print(f"  {'Modelo':<25} {'F1(0)':>7} {'Rec(0)':>7} {'Pre(0)':>7} {'F1mac':>7} {'AUC':>7}")
    print("-" * 65)
    for r in sorted(resultados, key=lambda x: x["f1_0_val"], reverse=True):
        marca = " ◄ GANADOR" if r["modelo"] == ganador["modelo"] else ""
        print(
            f"  {r['modelo']:<25} {r['f1_0_val']:>7.4f} {r['recall_0_val']:>7.4f} "
            f"{r['prec_0_val']:>7.4f} {r['f1_macro_val']:>7.4f} {r['auc_val']:>7.4f}{marca}"
        )
    print("=" * 80)

    duracion = round((datetime.now() - inicio).total_seconds(), 2)
    estado = {
        "script"         : "05_optuna",
        "timestamp"      : datetime.now().isoformat(),
        "duracion_seg"   : duracion,
        "n_trials"       : N_TRIALS,
        "pos_label"      : POS_LABEL,
        "seed"           : SEED,
        "modelo_ganador" : ganador["modelo"],
        "f1_0_val_ganador": ganador["f1_0_val"],
        "auc_val_ganador" : ganador["auc_val"],
        "f1_0_bt_ganador" : ganador["f1_0_bt"],
        "auc_bt_ganador"  : ganador["auc_bt"],
    }

    print("\n[5] Guardando resultados ...")
    guardar_resultados(resultados, estudios, modelos, estado)

    print(f"\n  Modelo ganador : {ganador['modelo']}")
    print(f"  F1(cls 0) val  : {ganador['f1_0_val']}")
    print(f"  AUC-ROC   val  : {ganador['auc_val']}")
    print(f"  F1(cls 0) bt   : {ganador['f1_0_bt']}")
    print(f"  Duración total : {duracion}s")
    print("Script 05 completado.\n")


if __name__ == "__main__":
    main()
