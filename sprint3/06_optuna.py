from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

ROOT      = Path(__file__).resolve().parent.parent
REPORTES  = ROOT / "data" / "reportes"
REPORTES.mkdir(parents=True, exist_ok=True)
DATA      = ROOT / "data" / "master"

SEED      = 2357
N_TRIALS  = 100
POS_LABEL = 0   # clase que nos importa: insatisfecho


# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────────────────────

def cargar_datos():
    X_train = pd.read_csv(DATA / "X_train.csv")
    X_val   = pd.read_csv(DATA / "X_val.csv")
    X_bt    = pd.read_csv(DATA / "X_backtest.csv")
    y_train = pd.read_csv(DATA / "y_train.csv").squeeze()
    y_val   = pd.read_csv(DATA / "y_val.csv").squeeze()
    y_bt    = pd.read_csv(DATA / "y_backtest.csv").squeeze()
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = round(pos / neg, 4)
    print(f"Train: {X_train.shape} | insatisfechos: {(y_train==0).mean():.2%}")
    print(f"Val:   {X_val.shape}   | insatisfechos: {(y_val==0).mean():.2%}")
    print(f"scale_pos_weight: {spw}")
    return X_train, X_val, X_bt, y_train, y_val, y_bt


# ─────────────────────────────────────────────────────────────────────────────
# Métricas comunes
# ─────────────────────────────────────────────────────────────────────────────

def metricas(y_true, y_pred, y_proba):
    return {
        "f1_0":      f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "recall_0":  recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "prec_0":    precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc":       roc_auc_score(y_true, y_proba),
        "accuracy":  float((y_true == y_pred).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Objective functions para Optuna
# ─────────────────────────────────────────────────────────────────────────────

def _make_objectives(X_train, y_train, X_val, y_val):

    def objective_lr(trial):
        params = {
            "C":            trial.suggest_float("C", 1e-3, 10, log=True),
            "solver":       trial.suggest_categorical("solver", ["lbfgs", "saga"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "max_iter":     1000,
            "random_state": SEED,
        }
        m = LogisticRegression(**params)
        m.fit(X_train, y_train)
        pred = (m.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_rf(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight":      trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state":      SEED,
            "n_jobs":            -2,
        }
        m = RandomForestClassifier(**params)
        m.fit(X_train, y_train)
        pred = (m.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_hgb(trial):
        params = {
            "max_iter":          trial.suggest_int("max_iter", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 2, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes", 15, 255),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-10, 1.0, log=True),
            "class_weight":      trial.suggest_categorical("class_weight", ["balanced", None]),
            "random_state":      SEED,
        }
        m = HistGradientBoostingClassifier(**params)
        m.fit(X_train, y_train)
        pred = (m.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_cb(trial):
        if not HAS_CATBOOST:
            raise optuna.exceptions.TrialPruned()
        params = {
            "iterations":         trial.suggest_int("iterations", 100, 1500),
            "depth":              trial.suggest_int("depth", 2, 10),
            "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "bagging_temperature":trial.suggest_float("bagging_temperature", 0.5, 1.0),
            "colsample_bylevel":  trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "l2_leaf_reg":        trial.suggest_float("l2_leaf_reg", 1e-3, 10, log=True),
            "auto_class_weights": trial.suggest_categorical("auto_class_weights", [None, "Balanced"]),
            "od_type":            trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
            "od_wait":            trial.suggest_int("od_wait", 10, 50),
            "random_state":       SEED,
            "verbose":            False,
            "task_type":          "CPU",
            "loss_function":      "Logloss",
        }
        m = CatBoostClassifier(**params)
        m.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        pred = (m.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_lgbm(trial):
        if not HAS_LGBM:
            raise optuna.exceptions.TrialPruned()
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
            "max_depth":       trial.suggest_int("max_depth", 2, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves":      trial.suggest_int("num_leaves", 20, 150),
            "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "class_weight":    trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state":    SEED,
            "verbosity":       -1,
            "n_jobs":          -2,
        }
        m = LGBMClassifier(**params)
        m.fit(X_train, y_train)
        pred = (m.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_lsvm(trial):
        p = {
            "C":            trial.suggest_float("C", 1e-3, 100.0, log=True),
            "loss":         trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
            "penalty":      "l2",
            "dual":         "auto",
            "tol":          trial.suggest_float("tol", 1e-5, 1e-1, log=True),
            "max_iter":     10000,
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state": SEED,
        }
        cal_method = trial.suggest_categorical("calibration_method", ["sigmoid", "isotonic"])
        m = CalibratedClassifierCV(LinearSVC(**p), method=cal_method, cv=5, n_jobs=-2)
        m.fit(X_train, y_train)
        pred = (m.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_sgd(trial):
        loss = trial.suggest_categorical("loss", ["log_loss", "modified_huber", "hinge"])
        lr   = trial.suggest_categorical("learning_rate", ["optimal", "constant", "invscaling", "adaptive"])
        p = {
            "loss":          loss,
            "penalty":       trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "alpha":         trial.suggest_float("alpha", 1e-5, 1.0, log=True),
            "learning_rate": lr,
            "max_iter":      2000,
            "tol":           trial.suggest_float("tol", 1e-5, 1e-2, log=True),
            "class_weight":  trial.suggest_categorical("class_weight", [None, "balanced"]),
            "n_jobs":        -2,
            "random_state":  SEED,
        }
        if lr != "optimal":
            p["eta0"] = trial.suggest_float("eta0", 1e-4, 0.1, log=True)
        if p["penalty"] == "elasticnet":
            p["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        m = SGDClassifier(**p)
        m.fit(X_train, y_train)
        if loss in ["log_loss", "modified_huber"]:
            proba = m.predict_proba(X_val)[:, 1]
        else:
            dec = m.decision_function(X_val)
            proba = 1 / (1 + np.exp(-dec))
        pred = (proba >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    def objective_et(trial):
        use_depth = trial.suggest_categorical("use_max_depth", [True, False])
        p = {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 50) if use_depth else None,
            "min_samples_split":trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.7]),
            "bootstrap":        trial.suggest_categorical("bootstrap", [False, True]),
            "class_weight":     trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
            "criterion":        trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "n_jobs":           -2,
            "random_state":     SEED,
        }
        if not p["bootstrap"] and p["class_weight"] == "balanced_subsample":
            p["class_weight"] = "balanced"
        m = ExtraTreesClassifier(**p)
        m.fit(X_train, y_train)
        pred = (m.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=POS_LABEL, zero_division=0)

    estudios_map = {
        "Logistic Regression":    objective_lr,
        "Random Forest":          objective_rf,
        "Hist Gradient Boosting": objective_hgb,
        "ExtraTrees":             objective_et,
        "SGD Classifier":         objective_sgd,
        "LinearSVM":              objective_lsvm,
    }
    if HAS_CATBOOST:
        estudios_map["CatBoost"] = objective_cb
    if HAS_LGBM:
        estudios_map["LightGBM"] = objective_lgbm

    return estudios_map


# ─────────────────────────────────────────────────────────────────────────────
# Reconstrucción de modelos desde best_params
# ─────────────────────────────────────────────────────────────────────────────

def construir_modelo(nombre, params):
    p = {k: v for k, v in params.items() if k not in ("use_max_depth", "calibration_method")}

    if nombre == "Logistic Regression":
        return LogisticRegression(**p, max_iter=1000, random_state=SEED)

    if nombre == "Random Forest":
        return RandomForestClassifier(**p, random_state=SEED, n_jobs=-2)

    if nombre == "Hist Gradient Boosting":
        return HistGradientBoostingClassifier(**p, random_state=SEED)

    if nombre == "CatBoost":
        return CatBoostClassifier(**p, random_state=SEED, loss_function="Logloss",
                                  verbose=False, task_type="CPU")

    if nombre == "LightGBM":
        return LGBMClassifier(**p, random_state=SEED, verbosity=-1, n_jobs=-2)

    if nombre == "LinearSVM":
        cal = params.get("calibration_method", "sigmoid")
        base = LinearSVC(**p, random_state=SEED, max_iter=10000)
        return CalibratedClassifierCV(base, method=cal, cv=5, n_jobs=-2)

    if nombre == "SGD Classifier":
        if p.get("learning_rate") == "optimal":
            p.pop("eta0", None)
        if p.get("loss") not in ["log", "log_loss", "modified_huber"]:
            p["loss"] = "log_loss"
        return SGDClassifier(**p, random_state=SEED, n_jobs=-1)

    if nombre == "ExtraTrees":
        return ExtraTreesClassifier(**p, random_state=SEED, n_jobs=-2)

    raise ValueError(f"Modelo desconocido: {nombre}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_inicio = time.time()
    print("=" * 55)
    print("06_optuna.py — Hiperparametrización Sprint 3")
    print("=" * 55)

    X_train, X_val, X_bt, y_train, y_val, y_bt = cargar_datos()
    estudios_map = _make_objectives(X_train, y_train, X_val, y_val)

    # ── 1. Ejecución de estudios Optuna ─────────────────────────────────────
    print(f"\n[1] Ejecutando Optuna ({N_TRIALS} trials por modelo)...\n")
    estudios_completados = {}
    for nombre, objective in estudios_map.items():
        print(f"  ▶ {nombre}")
        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            study_name=nombre,
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        estudios_completados[nombre] = study
        print(f"     Mejor F1(cls 0)={study.best_value:.4f}  t={time.time()-t0:.0f}s")
        print(f"     Params: {study.best_params}")

    # ── 2. CV 5-fold + OOF + re-entrenamiento ────────────────────────────────
    print("\n[2] 5-Fold CV + OOF + re-entrenamiento...\n")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    resultados_optuna = []
    oof_probas  = pd.DataFrame(index=X_train.index)
    test_probas = pd.DataFrame(index=X_val.index)

    for nombre, study in estudios_completados.items():
        print(f"  ▶ {nombre}")
        best_params = study.best_params.copy()

        oof_proba = np.zeros(len(X_train))
        cv_f1s = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr_f, X_val_f = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr_f, y_val_f = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            fold_model = construir_modelo(nombre, best_params)
            fold_model.fit(X_tr_f, y_tr_f)
            fp = fold_model.predict_proba(X_val_f)[:, 1]
            oof_proba[val_idx] = fp
            cv_f1s.append(f1_score(y_val_f, (fp >= 0.5).astype(int),
                                   pos_label=POS_LABEL, zero_division=0))

        oof_probas[nombre] = oof_proba
        cv_mean = float(np.mean(cv_f1s))
        cv_std  = float(np.std(cv_f1s))
        print(f"     CV F1: {cv_mean:.4f} (+/- {cv_std:.4f})")

        model_final = construir_modelo(nombre, best_params)
        model_final.fit(X_train, y_train)

        val_proba = model_final.predict_proba(X_val)[:, 1]
        test_probas[nombre] = val_proba
        m_val = metricas(y_val, (val_proba >= 0.5).astype(int), val_proba)

        bt_proba = model_final.predict_proba(X_bt)[:, 1]
        m_bt  = metricas(y_bt, (bt_proba >= 0.5).astype(int), bt_proba)

        resultados_optuna.append({
            "modelo":          nombre,
            "cv_f1_mean":      round(cv_mean, 4),
            "cv_f1_std":       round(cv_std, 4),
            "f1_0_val":        round(m_val["f1_0"], 4),
            "recall_0_val":    round(m_val["recall_0"], 4),
            "prec_0_val":      round(m_val["prec_0"], 4),
            "f1_macro_val":    round(m_val["f1_macro"], 4),
            "auc_val":         round(m_val["auc"], 4),
            "accuracy_val":    round(m_val["accuracy"], 4),
            "f1_0_bt":         round(m_bt["f1_0"], 4),
            "recall_0_bt":     round(m_bt["recall_0"], 4),
            "prec_0_bt":       round(m_bt["prec_0"], 4),
            "f1_macro_bt":     round(m_bt["f1_macro"], 4),
            "auc_bt":          round(m_bt["auc"], 4),
            "accuracy_bt":     round(m_bt["accuracy"], 4),
            "best_params":     best_params,
            "_proba_val":      val_proba,
            "_proba_bt":       bt_proba,
            "_model":          model_final,
        })

    # ── 3. Stacking manual con OOF ────────────────────────────────────────────
    print("\n[3] Stacking manual (meta-learner LogReg)...")
    meta_learner = LogisticRegression(C=1.0, class_weight="balanced",
                                      max_iter=1000, random_state=SEED)
    meta_learner.fit(oof_probas, y_train)

    stack_proba = meta_learner.predict_proba(test_probas)[:, 1]
    m_stack = metricas(y_val, (stack_proba >= 0.5).astype(int), stack_proba)
    print(f"  Stacking → F1(cls0)={m_stack['f1_0']:.4f}  AUC={m_stack['auc']:.4f}")

    # ── 4. Persistencia ───────────────────────────────────────────────────────
    print("\n[4] Guardando artefactos...")

    modelos_dict = {r["modelo"]: r["_model"] for r in resultados_optuna}
    joblib.dump(modelos_dict, REPORTES / "modelos_optuna.pkl")
    joblib.dump(estudios_completados, REPORTES / "estudios_optuna.pkl")
    joblib.dump(meta_learner, REPORTES / "meta_learner.pkl")
    joblib.dump(oof_probas, REPORTES / "oof_probas.pkl")
    joblib.dump(test_probas, REPORTES / "test_probas.pkl")

    mejores_params = {r["modelo"]: r["best_params"] for r in resultados_optuna}
    with open(REPORTES / "mejores_params.json", "w") as f:
        json.dump(mejores_params, f, indent=2, default=str)

    registros_json = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in resultados_optuna
    ]
    with open(REPORTES / "resultados_optuna.json", "w") as f:
        json.dump(registros_json, f, indent=2)

    ganador = max(resultados_optuna, key=lambda x: x["f1_0_val"])
    duracion = round(time.time() - t_inicio, 2)
    estado = {
        "script":           "06_optuna",
        "timestamp":        datetime.now().isoformat(),
        "duracion_seg":     duracion,
        "n_trials":         N_TRIALS,
        "pos_label":        POS_LABEL,
        "seed":             SEED,
        "modelo_ganador":   ganador["modelo"],
        "f1_0_val_ganador": ganador["f1_0_val"],
        "auc_val_ganador":  ganador["auc_val"],
        "f1_0_bt_ganador":  ganador["f1_0_bt"],
        "auc_bt_ganador":   ganador["auc_bt"],
        "stacking_f1_0":    round(m_stack["f1_0"], 4),
        "stacking_auc":     round(m_stack["auc"], 4),
    }
    with open(REPORTES / "estado_optuna.json", "w") as f:
        json.dump(estado, f, indent=2)

    print(f"\n✅ 06_optuna.py completado en {duracion}s")
    print(f"   Modelo ganador: {ganador['modelo']}  F1(cls0)={ganador['f1_0_val']}  AUC={ganador['auc_val']}")
    print(f"   → data/reportes/resultados_optuna.json")
    print(f"   → data/reportes/mejores_params.json")
    print(f"   → data/reportes/modelos_optuna.pkl")
    print(f"   → data/reportes/estado_optuna.json")


if __name__ == "__main__":
    main()