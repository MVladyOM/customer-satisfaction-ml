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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

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
        # Clase 0 — Insatisfecho
        "f1_0":     f1_score(y_true, y_pred,        pos_label=0, zero_division=0),
        "recall_0": recall_score(y_true, y_pred,    pos_label=0, zero_division=0),
        "prec_0":   precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        # Clase 1 — Satisfecho
        "f1_1":     f1_score(y_true, y_pred,        pos_label=1, zero_division=0),
        "recall_1": recall_score(y_true, y_pred,    pos_label=1, zero_division=0),
        "prec_1":   precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        # General
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc":      roc_auc_score(y_true, y_proba),
        "accuracy": float((y_true == y_pred).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Objective functions para Optuna
# ─────────────────────────────────────────────────────────────────────────────

def _make_objectives(X_train, y_train, X_val, y_val, pos_label: int = 0):

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
        return f1_score(y_val, pred, pos_label=pos_label, zero_division=0)

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
        return f1_score(y_val, pred, pos_label=pos_label, zero_division=0)

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
        m = CalibratedClassifierCV(LinearSVC(**p), method=cal_method, cv=5)
        m.fit(X_train, y_train)
        pred = (m.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        return f1_score(y_val, pred, pos_label=pos_label, zero_division=0)

    return {
        "Logistic Regression":    objective_lr,
        "Hist Gradient Boosting": objective_hgb,
        "LinearSVM":              objective_lsvm,
    }


def _ejecutar_ronda(estudios_map, X_train, X_val, X_bt,
                    y_train, y_val, y_bt, skf, pos_label: int):
    """Corre Optuna + CV 5-fold + re-entrenamiento para un pos_label dado."""
    cls_tag = f"cls{pos_label}"

    # ── Estudios Optuna ───────────────────────────────────────────────────────
    estudios = {}
    for nombre, objective in estudios_map.items():
        print(f"  ▶ {nombre}")
        t0 = time.time()
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            study_name=f"{nombre}_{cls_tag}",
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        estudios[nombre] = study
        print(f"     Mejor F1({cls_tag})={study.best_value:.4f}  t={time.time()-t0:.0f}s")
        print(f"     Params: {study.best_params}")

    # ── CV 5-fold + OOF + re-entrenamiento ───────────────────────────────────
    print(f"\n  CV 5-fold + re-entrenamiento ({cls_tag})...\n")
    resultados = []
    oof_probas  = pd.DataFrame(index=X_train.index)
    test_probas = pd.DataFrame(index=X_val.index)

    for nombre, study in estudios.items():
        print(f"  ▶ {nombre}")
        best_params = study.best_params.copy()

        oof_proba = np.zeros(len(X_train))
        cv_f1s = []

        for _, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr_f, X_val_f = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr_f, y_val_f = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            fold_model = construir_modelo(nombre, best_params)
            fold_model.fit(X_tr_f, y_tr_f)
            fp = fold_model.predict_proba(X_val_f)[:, 1]
            oof_proba[val_idx] = fp
            cv_f1s.append(f1_score(y_val_f, (fp >= 0.5).astype(int),
                                   pos_label=pos_label, zero_division=0))

        oof_probas[nombre] = oof_proba
        cv_mean = float(np.mean(cv_f1s))
        cv_std  = float(np.std(cv_f1s))
        print(f"     CV F1({cls_tag}): {cv_mean:.4f} (+/- {cv_std:.4f})")

        model_final = construir_modelo(nombre, best_params)
        model_final.fit(X_train, y_train)

        val_proba = model_final.predict_proba(X_val)[:, 1]
        test_probas[nombre] = val_proba
        m_val = metricas(y_val, (val_proba >= 0.5).astype(int), val_proba)

        bt_proba = model_final.predict_proba(X_bt)[:, 1]
        m_bt = metricas(y_bt, (bt_proba >= 0.5).astype(int), bt_proba)

        resultados.append({
            "modelo":          nombre,
            "pos_label":       pos_label,
            "cv_f1_mean":      round(cv_mean, 4),
            "cv_f1_std":       round(cv_std, 4),
            "f1_0_val":        round(m_val["f1_0"], 4),
            "recall_0_val":    round(m_val["recall_0"], 4),
            "prec_0_val":      round(m_val["prec_0"], 4),
            "f1_1_val":        round(m_val["f1_1"], 4),
            "recall_1_val":    round(m_val["recall_1"], 4),
            "prec_1_val":      round(m_val["prec_1"], 4),
            "f1_macro_val":    round(m_val["f1_macro"], 4),
            "auc_val":         round(m_val["auc"], 4),
            "accuracy_val":    round(m_val["accuracy"], 4),
            "f1_0_bt":         round(m_bt["f1_0"], 4),
            "recall_0_bt":     round(m_bt["recall_0"], 4),
            "prec_0_bt":       round(m_bt["prec_0"], 4),
            "f1_1_bt":         round(m_bt["f1_1"], 4),
            "recall_1_bt":     round(m_bt["recall_1"], 4),
            "prec_1_bt":       round(m_bt["prec_1"], 4),
            "f1_macro_bt":     round(m_bt["f1_macro"], 4),
            "auc_bt":          round(m_bt["auc"], 4),
            "accuracy_bt":     round(m_bt["accuracy"], 4),
            "best_params":     best_params,
            "_proba_val":      val_proba,
            "_proba_bt":       bt_proba,
            "_model":          model_final,
        })

    return estudios, resultados, oof_probas, test_probas


# ─────────────────────────────────────────────────────────────────────────────
# Reconstrucción de modelos desde best_params
# ─────────────────────────────────────────────────────────────────────────────

def construir_modelo(nombre, params):
    p = {k: v for k, v in params.items() if k not in ("use_max_depth", "calibration_method")}

    if nombre == "Logistic Regression":
        return LogisticRegression(**p, max_iter=1000, random_state=SEED)

    if nombre == "Hist Gradient Boosting":
        return HistGradientBoostingClassifier(**p, random_state=SEED)

    if nombre == "LinearSVM":
        cal = params.get("calibration_method", "sigmoid")
        base = LinearSVC(**p, random_state=SEED, max_iter=10000)
        return CalibratedClassifierCV(base, method=cal, cv=5)

    raise ValueError(f"Modelo desconocido: {nombre}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def _guardar_ronda(resultados, estudios, oof_probas, test_probas,
                   y_train, y_val, pos_label: int, duracion: float):
    """Persiste artefactos de una ronda Optuna. Sufijo _cls0 / _cls1."""
    tag  = f"cls{pos_label}"
    mdir = REPORTES / "modelos"
    mdir.mkdir(exist_ok=True)

    modelos_dict = {r["modelo"]: r["_model"] for r in resultados}
    joblib.dump(modelos_dict,  REPORTES / f"modelos_optuna_{tag}.pkl")
    joblib.dump(estudios,      REPORTES / f"estudios_optuna_{tag}.pkl")
    joblib.dump(oof_probas,    REPORTES / f"oof_probas_{tag}.pkl")
    joblib.dump(test_probas,   REPORTES / f"test_probas_{tag}.pkl")

    for r in resultados:
        slug = r["modelo"].lower().replace(" ", "_")
        joblib.dump(r["_model"], mdir / f"modelo_{slug}_{tag}.pkl")

    mejores = {r["modelo"]: r["best_params"] for r in resultados}
    with open(REPORTES / f"mejores_params_{tag}.json", "w") as f:
        json.dump(mejores, f, indent=2, default=str)

    registros = [{k: v for k, v in r.items() if not k.startswith("_")}
                 for r in resultados]
    with open(REPORTES / f"resultados_optuna_{tag}.json", "w") as f:
        json.dump(registros, f, indent=2)

    # Stacking con OOF de esta ronda
    meta = LogisticRegression(C=1.0, class_weight="balanced",
                               max_iter=1000, random_state=SEED)
    meta.fit(oof_probas, y_train)
    stack_proba = meta.predict_proba(test_probas)[:, 1]
    m_stack = metricas(y_val, (stack_proba >= 0.5).astype(int), stack_proba)
    joblib.dump(meta, REPORTES / f"meta_learner_{tag}.pkl")
    print(f"  Stacking {tag} → F1(cls0)={m_stack['f1_0']:.4f}  "
          f"F1(cls1)={m_stack['f1_1']:.4f}  AUC={m_stack['auc']:.4f}")

    ganador_key = f"f1_{pos_label}_val"
    ganador = max(resultados, key=lambda x: x[ganador_key])
    print(f"  Ganador {tag}: {ganador['modelo']}  "
          f"F1({tag})={ganador[ganador_key]}  AUC={ganador['auc_val']}")
    return ganador, m_stack


def main():
    t_inicio = time.time()
    print("=" * 55)
    print("06_optuna.py — Hiperparametrización Sprint 3")
    print("=" * 55)

    X_train, X_val, X_bt, y_train, y_val, y_bt = cargar_datos()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ── Ronda 1: optimizar F1 clase 0 (insatisfecho) ─────────────────────────
    print(f"\n[1] Optuna — optimizando F1 cls 0  ({N_TRIALS} trials/modelo)...\n")
    map_cls0 = _make_objectives(X_train, y_train, X_val, y_val, pos_label=0)
    est0, res0, oof0, test0 = _ejecutar_ronda(
        map_cls0, X_train, X_val, X_bt, y_train, y_val, y_bt, skf, pos_label=0
    )

    # ── Ronda 2: optimizar F1 clase 1 (satisfecho) ───────────────────────────
    print(f"\n[2] Optuna — optimizando F1 cls 1  ({N_TRIALS} trials/modelo)...\n")
    map_cls1 = _make_objectives(X_train, y_train, X_val, y_val, pos_label=1)
    est1, res1, oof1, test1 = _ejecutar_ronda(
        map_cls1, X_train, X_val, X_bt, y_train, y_val, y_bt, skf, pos_label=1
    )

    # ── Persistencia ─────────────────────────────────────────────────────────
    print("\n[3] Guardando artefactos...\n")
    duracion = round(time.time() - t_inicio, 2)

    gan0, stack0 = _guardar_ronda(res0, est0, oof0, test0, y_train, y_val, 0, duracion)
    gan1, stack1 = _guardar_ronda(res1, est1, oof1, test1, y_train, y_val, 1, duracion)

    # Para compatibilidad con 07_graficos.py — cls0 como principal
    joblib.dump({r["modelo"]: r["_model"] for r in res0},
                REPORTES / "modelos_optuna.pkl")
    joblib.dump(est0, REPORTES / "estudios_optuna.pkl")
    with open(REPORTES / "resultados_optuna.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if not k.startswith("_")}
                   for r in res0], f, indent=2)
    with open(REPORTES / "mejores_params.json", "w") as f:
        json.dump({r["modelo"]: r["best_params"] for r in res0}, f, indent=2, default=str)

    estado = {
        "script":              "06_optuna",
        "timestamp":           datetime.now().isoformat(),
        "duracion_seg":        duracion,
        "n_trials":            N_TRIALS,
        "seed":                SEED,
        "ganador_cls0":        gan0["modelo"],
        "f1_0_val_cls0":       gan0["f1_0_val"],
        "auc_val_cls0":        gan0["auc_val"],
        "ganador_cls1":        gan1["modelo"],
        "f1_1_val_cls1":       gan1["f1_1_val"],
        "auc_val_cls1":        gan1["auc_val"],
        "stacking_cls0_f1_0":  round(stack0["f1_0"], 4),
        "stacking_cls0_f1_1":  round(stack0["f1_1"], 4),
        "stacking_cls0_auc":   round(stack0["auc"],  4),
        "stacking_cls1_f1_0":  round(stack1["f1_0"], 4),
        "stacking_cls1_f1_1":  round(stack1["f1_1"], 4),
        "stacking_cls1_auc":   round(stack1["auc"],  4),
    }
    with open(REPORTES / "estado_optuna.json", "w") as f:
        json.dump(estado, f, indent=2)

    print(f"\n✅ 06_optuna.py completado en {duracion}s")
    print(f"   → data/reportes/resultados_optuna_cls0.json  /  resultados_optuna_cls1.json")
    print(f"   → data/reportes/mejores_params_cls0.json     /  mejores_params_cls1.json")
    print(f"   → data/reportes/modelos_optuna_cls0.pkl      /  modelos_optuna_cls1.pkl")
    print(f"   → data/reportes/modelos/modelo_<nombre>_cls0.pkl  /  _cls1.pkl")
    print(f"   → data/reportes/estado_optuna.json")


if __name__ == "__main__":
    main()