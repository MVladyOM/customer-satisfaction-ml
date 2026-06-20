"""
=============================================================================
Script 04 - Split Temporal + Transformaciones + Selección de Variables
Proyecto : Predicción de Satisfacción del Cliente — Olist
Grupo 1  | Sprint 2
=============================================================================
Responsabilidad:
  1. Separar X / y
  2. Split temporal estricto (sin shuffle):
       Train    : sep 2016 – dic 2017  (16 meses)
       Val      : ene 2018 – mar 2018  ( 3 meses)
       Backtest : abr 2018 – jun 2018  ( 3 meses)
       Live     : jul 2018 – oct 2018  ( resto  )
  3. Fitear transformadores SOLO en Train → aplicar al resto:
       - Target Encoding (media del target por categoría, fit en train)
       - StandardScaler  (fit en train)
  4. Selección de variables por pasos (SOLO train + val):

     LÓGICA DE EVALUACIÓN:
       Cada paso recibe el conjunto de features del paso anterior.
       Dentro de cada paso se prueban TODOS los umbrales en PARALELO
       (todos parten del mismo conjunto base). Se elige el umbral con
       mejor AUC_val, siempre que no deje 0 features ni degrade el
       AUC_val más de TOLERANCIA_AUC respecto al estado inicial del paso.
       Si ningún umbral mejora o todos dejan 0 features, el paso se
       omite y se conserva el conjunto base sin cambios.

       Paso 0 → estado inicial
       Paso 1 → missing_variable_method   (umbrales: 0.10, 0.15)
       Paso 2 → PSI_method                (bins=10, umbral PSI > 0.20)
       Paso 3 → correlation_method        (umbrales: 0.80, 0.90, 0.95, 0.99)
       Paso 4 → univariante_method        (umbrales: 0.10, 0.20, 0.30)
       Paso 5 → variance_threshold_method (umbral: 0.01)      ← EXTRA 1
       Paso 6 → rf_importance_method      (top-N)             ← EXTRA 2

  5. Guardar splits finales + reportes

REGLA DE ORO: Backtest y Live NUNCA se tocan durante la selección.
              Solo reciben los transformadores ya fiteados.

Entrada : data/master/master_table_limpia.csv
Salida  : data/master/X_train.csv / y_train.csv
          data/master/X_val.csv   / y_val.csv
          data/master/X_backtest.csv / y_backtest.csv
          data/master/X_live.csv     / y_live.csv
          data/reportes/features_seleccionadas.json
          data/reportes/tabla_seleccion_variables.csv
          data/reportes/tabla_detalle_eliminadas.csv
          data/reportes/estado_seleccion.json
=============================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

# Forzar UTF-8 en stdout para compatibilidad con Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
MASTER_PATH = os.path.join("data", "master")
REPORTES_PATH = os.path.join("data", "reportes")

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
FECHA_FIN_TRAIN = "2017-12-31"
FECHA_FIN_VAL = "2018-03-31"
FECHA_FIN_BACKTEST = "2018-06-30"

# Umbrales múltiples — cada método los prueba en PARALELO
UMBRALES_MISSINGS = [0.10, 0.15]
UMBRAL_PSI = 0.20
BINS_PSI = 10
UMBRALES_CORRELACION = [0.80, 0.90, 0.95, 0.99]
# UMBRALES_UNIVARIANTE  = [0.10, 0.20, 0.30]
UMBRALES_UNIVARIANTE = [0.01, 0.02, 0.05]
UMBRAL_VARIANZA = 0.01          # Variance Threshold (EXTRA 1)
# RF Importance — probar varios top-N (EXTRA 2)
RF_IMPORTANCE_TOP_NS = [20, 25, 30]

# Cuánto se permite degradar AUC_val al seleccionar features
# Si el mejor umbral baja el AUC_val más de esto, el paso se omite
TOLERANCIA_AUC = 0.005   # 0.5 pp — margen mínimo aceptable

# Modelo de evaluación rápida (NO es el modelo final)
RF_PARAMS = dict(
    n_estimators=100, max_depth=6, n_jobs=-1,
    class_weight="balanced", random_state=42
)

# Target Encoding
COLS_TE = [
    "product_category_name_english"
]

COL_FECHA = "order_purchase_timestamp"


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------

def cargar_master() -> pd.DataFrame:
    ruta = os.path.join(MASTER_PATH, "master_table_limpia.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró {ruta}. Ejecuta primero el Script 03."
        )
    df = pd.read_csv(ruta, low_memory=False)
    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce")

    print(f"  Master limpia cargada: {df.shape[0]:,} filas × {df.shape[1]} columnas\n")

    # ── Reporte de columnas y nulos ──────────────────────────────────────────
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)

    ancho_col  = max(len(c) for c in df.columns) + 2
    print(f"  {'#':<5} {'Columna':<{ancho_col}} {'Dtype':<15} {'Nulos':>8} {'%Nulos':>8}  Estado")
    print("  " + "-" * (ancho_col + 45))

    cols_con_nulos = 0
    for i, col in enumerate(df.columns, 1):
        n_null = null_counts[col]
        pct    = null_pct[col]
        estado = "OK" if n_null == 0 else f"⚠  {pct:.2f}% nulos"
        if n_null > 0:
            cols_con_nulos += 1
        print(f"  {i:<5} {col:<{ancho_col}} {str(df[col].dtype):<15} {n_null:>8,} {pct:>7.2f}%  {estado}")

    print("  " + "-" * (ancho_col + 45))
    total_nulos = null_counts.sum()
    print(f"\n  Resumen nulos:")
    print(f"    Columnas con nulos : {cols_con_nulos} / {df.shape[1]}")
    print(f"    Total celdas nulas : {total_nulos:,}  "
          f"({total_nulos / df.size * 100:.3f}% del total)")
    if cols_con_nulos == 0:
        print("    Sin valores nulos detectados.")
    print()

    # ── Corrección automática de nulos ──────────────────────────────────────
    if total_nulos > 0:
        print("  Corrigiendo nulos detectados:")
        filas_antes = len(df)

        for col in df.columns:
            n = null_counts[col]
            if n == 0:
                continue

            if col == COL_FECHA:
                # Fecha requerida para el split temporal — eliminar esas filas
                df.dropna(subset=[col], inplace=True)
                eliminadas = filas_antes - len(df)
                print(f"    {col:<{ancho_col}} → eliminadas {eliminadas:,} filas "
                      f"(fecha nula impide split temporal)")

            elif pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                print(f"    {col:<{ancho_col}} → {n:,} nulos imputados con mediana "
                      f"= {fill_val:.4f}")

            else:
                moda = df[col].mode()
                fill_val = moda.iloc[0] if not moda.empty else "Unknown"
                df[col].fillna(fill_val, inplace=True)
                print(f"    {col:<{ancho_col}} → {n:,} nulos imputados con moda "
                      f"= '{fill_val}'")

        nulos_restantes = df.isnull().sum().sum()
        print(f"\n    Nulos restantes tras corrección : {nulos_restantes}")
        if len(df) < filas_antes:
            print(f"    Filas eliminadas (fecha nula)   : {filas_antes - len(df):,}")
        print(f"    Shape final                     : {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print()

    return df


# ---------------------------------------------------------------------------
# 2. SPLIT TEMPORAL
# ---------------------------------------------------------------------------

def split_temporal(df: pd.DataFrame) -> tuple:
    mask_train = df[COL_FECHA] <= FECHA_FIN_TRAIN
    mask_val = (df[COL_FECHA] > FECHA_FIN_TRAIN) & (
        df[COL_FECHA] <= FECHA_FIN_VAL)
    mask_backtest = (df[COL_FECHA] > FECHA_FIN_VAL) & (
        df[COL_FECHA] <= FECHA_FIN_BACKTEST)
    mask_live = df[COL_FECHA] > FECHA_FIN_BACKTEST

    train    = df[mask_train].copy().reset_index(drop=True)
    val      = df[mask_val].copy().reset_index(drop=True)
    backtest = df[mask_backtest].copy().reset_index(drop=True)
    live     = df[mask_live].copy().reset_index(drop=True)

    for nombre, parte in [("Train", train), ("Val", val),
                          ("Backtest", backtest), ("Live", live)]:
        fecha_min = parte[COL_FECHA].min().strftime("%Y-%m")
        fecha_max = parte[COL_FECHA].max().strftime("%Y-%m")
        pct_sat = parte["satisfecho"].mean()
        print(f"  {nombre:10s}: {len(parte):>7,} filas  "
              f"({fecha_min} → {fecha_max})  satisfecho={pct_sat:.1%}")

    return train, val, backtest, live


def separar_Xy(df: pd.DataFrame) -> tuple:
    y = df["satisfecho"].copy()
    X = df.drop(columns=["satisfecho", COL_FECHA, "order_id"], errors="ignore").copy()
    return X, y


# ---------------------------------------------------------------------------
# 3. TRANSFORMACIONES (fit SOLO en train)
# ---------------------------------------------------------------------------

def aplicar_target_encoding(
    X_train, y_train, X_val, X_backtest, X_live
) -> tuple:
    global_mean = float(y_train.mean())
    te_maps = {}
    for col in COLS_TE:
        if col not in X_train.columns:
            continue
        mapping = (
            X_train[[col]]
            .assign(target=y_train.values)
            .groupby(col)["target"]
            .mean()
            .to_dict()
        )
        te_maps[col] = mapping
        nuevo_col = f"te_{col}"
        for df_part in [X_train, X_val, X_backtest, X_live]:
            df_part[nuevo_col] = df_part[col].map(mapping).fillna(global_mean)
            df_part.drop(columns=[col], inplace=True)
        print(f"  te_{col} → {len(mapping)} categorías mapeadas")
    return X_train, X_val, X_backtest, X_live, te_maps


def aplicar_scaler(X_train, X_val, X_backtest, X_live) -> tuple:
    cols_num = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[cols_num] = scaler.fit_transform(X_train[cols_num])
    for df_part in [X_val, X_backtest, X_live]:
        df_part[cols_num] = scaler.transform(df_part[cols_num])
    print(
        f"  StandardScaler aplicado sobre {len(cols_num)} columnas numéricas")
    return X_train, X_val, X_backtest, X_live, scaler, cols_num


# ---------------------------------------------------------------------------
# 4. UTILIDAD: ENTRENAR RF Y MEDIR TODAS LAS MÉTRICAS
# ---------------------------------------------------------------------------

def evaluar_rf_completo(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_vl: pd.DataFrame, y_vl: pd.Series,
    features: list,
    threshold: float = 0.5
) -> dict:
    """
    Entrena RF rápido. Devuelve dict con TODAS las métricas o None si falla.
    """
    if not features:
        return {
            "auc_train": None, "auc_val": None,
            "acc_train": None, "acc_val": None,
            "rec_train": None, "rec_val": None,
            "prec_train": None, "prec_val": None,
            "f1_train": None, "f1_val": None,
            "gini_train": None, "gini_val": None,
        }
    # Filtrar features con varianza > 0 en train
    valid = [f for f in features
             if f in X_tr.columns and X_tr[f].nunique() > 1]
    if not valid:
        return {
            "auc_train": None, "auc_val": None,
            "acc_train": None, "acc_val": None,
            "rec_train": None, "rec_val": None,
            "prec_train": None, "prec_val": None,
            "f1_train": None, "f1_val": None,
            "gini_train": None, "gini_val": None,
        }
    X_tr_f = X_tr[valid].astype(float)
    X_vl_f = X_vl[valid].astype(float)
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_tr_f, y_tr)

    # Probabilidades
    prob_tr = rf.predict_proba(X_tr_f)[:, 1]
    prob_vl = rf.predict_proba(X_vl_f)[:, 1]

    # Predicciones con threshold
    pred_tr = (prob_tr >= threshold).astype(int)
    pred_vl = (prob_vl >= threshold).astype(int)

    # Funciones auxiliares
    def safe_roc_auc(y_true, y_prob):
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)

    def gini(auc):
        return 2 * auc - 1

    # Calcular métricas
    auc_tr = safe_roc_auc(y_tr, prob_tr)
    auc_vl = safe_roc_auc(y_vl, prob_vl)

    return {
        "auc_train":  round(auc_tr, 4) if not np.isnan(auc_tr) else None,
        "auc_val":    round(auc_vl, 4) if not np.isnan(auc_vl) else None,
        "acc_train":  round(accuracy_score(y_tr, pred_tr), 4),
        "acc_val":    round(accuracy_score(y_vl, pred_vl), 4),
        "rec_train":  round(recall_score(y_tr, pred_tr, zero_division=0), 4),
        "rec_val":    round(recall_score(y_vl, pred_vl, zero_division=0), 4),
        "prec_train": round(precision_score(y_tr, pred_tr, zero_division=0), 4),
        "prec_val":   round(precision_score(y_vl, pred_vl, zero_division=0), 4),
        "f1_train":   round(f1_score(y_tr, pred_tr, zero_division=0), 4),
        "f1_val":     round(f1_score(y_vl, pred_vl, zero_division=0), 4),
        "gini_train": round(gini(auc_tr), 4) if not np.isnan(auc_tr) else None,
        "gini_val":   round(gini(auc_vl), 4) if not np.isnan(auc_vl) else None,
    }


def evaluar_rf(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_vl: pd.DataFrame, y_vl: pd.Series,
    features: list
) -> tuple:
    """Wrapper legacy: devuelve (AUC_train, AUC_val) o (None, None)."""
    metricas = evaluar_rf_completo(X_tr, y_tr, X_vl, y_vl, features)
    return metricas["auc_train"], metricas["auc_val"]


# ---------------------------------------------------------------------------
# 5. FILTROS INDIVIDUALES (devuelven candidatos, no modifican estado global)
# ---------------------------------------------------------------------------

def filtrar_missings(X_train, features, umbral):
    pct = X_train[features].isnull().mean()
    eliminar = pct[pct > umbral].index.tolist()
    return [f for f in features if f not in eliminar], eliminar


def calcular_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    exp_clean = expected.dropna()
    act_clean = actual.dropna()
    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0
    breakpoints = np.unique(np.percentile(
        exp_clean, np.linspace(0, 100, bins + 1)))
    if len(breakpoints) < 2:
        return 0.0
    exp_cnt = np.histogram(exp_clean, bins=breakpoints)[0]
    act_cnt = np.histogram(act_clean, bins=breakpoints)[0]
    exp_pct = np.where(exp_cnt == 0, 1e-6, exp_cnt / len(exp_clean))
    act_pct = np.where(act_cnt == 0, 1e-6, act_cnt / len(act_clean))
    return round(float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))), 4)


def filtrar_psi(X_train, X_val, features, umbral=UMBRAL_PSI):
    psi_vals = {col: calcular_psi(X_train[col], X_val[col], BINS_PSI)
                for col in features}
    eliminar = [col for col, psi in psi_vals.items() if psi > umbral]
    return [f for f in features if f not in eliminar], psi_vals, eliminar


def filtrar_correlacion(X_train, features, umbral):
    if len(features) < 2:
        return features, []
    corr_matrix = X_train[features].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    eliminar = [col for col in upper.columns if any(upper[col] > umbral)]
    return [f for f in features if f not in eliminar], eliminar


def filtrar_univariante(X_train, y_train, X_val, y_val, features, umbral):
    """
    Para cada feature entrena RF univariante en train y evalúa AUC en val.
    Elimina si |AUC_val - 0.5| < umbral  (sin poder predictivo individual).
    """
    aucs = {}
    eliminar = []
    for col in features:
        try:
            if X_train[col].nunique() <= 1:
                eliminar.append(col)
                aucs[col] = None
                continue
            rf_uni = RandomForestClassifier(
                n_estimators=30, max_depth=3,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
            rf_uni.fit(X_train[[col]].astype(float), y_train)
            auc = roc_auc_score(y_val,
                                rf_uni.predict_proba(X_val[[col]].astype(float))[:, 1])
            aucs[col] = round(float(auc), 4)
            if abs(auc - 0.5) < umbral:
                eliminar.append(col)
        except Exception:
            eliminar.append(col)
            aucs[col] = None
    return [f for f in features if f not in eliminar], aucs, eliminar


def filtrar_varianza(X_train, features, umbral=UMBRAL_VARIANZA):
    if not features:
        return features, []
    X_num = X_train[features].astype(float).fillna(0)
    selector = VarianceThreshold(threshold=umbral)
    selector.fit(X_num)
    mask = selector.get_support()
    ok = [f for f, m in zip(features, mask) if m]
    elim = [f for f, m in zip(features, mask) if not m]
    return ok, elim


def filtrar_rf_importance(X_train, y_train, features, top_n):
    if not features or len(features) <= top_n:
        return features, {}, []
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train[features].astype(float), y_train)
    importancias = dict(zip(features, rf.feature_importances_))
    ord_desc = sorted(importancias.items(), key=lambda x: x[1], reverse=True)
    ok = [f for f, _ in ord_desc[:top_n]]
    elim = [f for f, _ in ord_desc[top_n:]]
    return ok, importancias, elim


# ---------------------------------------------------------------------------
# 6. ORQUESTADOR — evalúa umbrales en PARALELO, elige el mejor
# ---------------------------------------------------------------------------

def _fmt_auc(v):
    return f"{v:.4f}" if isinstance(v, float) else "  N/A "


def seleccion_de_variables(
    X_train, y_train, X_val, y_val
) -> tuple:
    """
    Cada paso:
      1. Parte del conjunto de features aprobado en el paso anterior (base).
      2. Prueba TODOS sus umbrales/variantes en PARALELO sobre ese mismo base.
      3. Registra cada candidato en la tabla de resultados.
      4. Elige el candidato con MAYOR AUC_val, siempre que:
           - deje al menos 1 feature
           - no degrade AUC_val más de TOLERANCIA_AUC vs el AUC_val del base
      5. Si ningún candidato cumple, el paso se omite (se conserva base).
      6. El ganador pasa como nuevo base al siguiente paso.
    """
    tabla = []   # filas para el CSV/consola
    features = X_train.columns.tolist()

    # ── utilidad interna ──────────────────────────────────────────────────
    def registrar(paso, metodo, umbral_str, features_cand,
                  eliminadas, es_ganador=False):
        metricas = evaluar_rf_completo(
            X_train, y_train, X_val, y_val, features_cand)
        auc_tr = metricas["auc_train"]
        auc_vl = metricas["auc_val"]
        marca = " ◄ ELEGIDO" if es_ganador else ""
        elim_str = f"(-{len(eliminadas)})" if eliminadas else ""
        print(
            f"  {'Paso':4} {paso:<2} | {metodo:<36} umbral={umbral_str:<14} | "
            f"{len(features_cand):>3} feat | "
            f"AUC_tr={_fmt_auc(auc_tr)}  AUC_vl={_fmt_auc(auc_vl)}  "
            f"{elim_str}{marca}"
        )
        if eliminadas and es_ganador:
            print(f"         └─ Eliminadas: {', '.join(eliminadas)}")
        tabla.append({
            "paso": paso,
            "metodo": metodo,
            "threshold": umbral_str,
            "n_features": len(features_cand),
            "auc_train": auc_tr,
            "auc_val": auc_vl,
            "acc_train": metricas["acc_train"],
            "acc_val": metricas["acc_val"],
            "rec_train": metricas["rec_train"],
            "rec_val": metricas["rec_val"],
            "prec_train": metricas["prec_train"],
            "prec_val": metricas["prec_val"],
            "f1_train": metricas["f1_train"],
            "f1_val": metricas["f1_val"],
            "gini_train": metricas["gini_train"],
            "gini_val": metricas["gini_val"],
            "n_eliminadas": len(eliminadas),
            "features_eliminadas": eliminadas,
            "elegido": es_ganador,
        })
        return auc_tr, auc_vl

    def elegir_mejor(candidatos, auc_val_base):
        """
        candidatos: list of (features_cand, eliminadas, umbral_str, auc_vl)
        Devuelve el candidato con mayor auc_vl que cumpla los criterios,
        o None si ninguno cumple.
        """
        validos = [
            c for c in candidatos
            if c[3] is not None          # tiene AUC calculable
            and len(c[0]) > 0            # deja al menos 1 feature
            # no degrada más de tolerancia
            and c[3] >= auc_val_base - TOLERANCIA_AUC
        ]
        if not validos:
            return None
        return max(validos, key=lambda c: c[3])

    # ── Paso 0: Estado inicial ─────────────────────────────────────────────
    metricas0 = evaluar_rf_completo(X_train, y_train, X_val, y_val, features)
    auc_tr0 = metricas0["auc_train"]
    auc_vl0 = metricas0["auc_val"]
    print(f"  Paso 0 | Estado_Inicial"
          f"{'':>42} | {len(features):>3} feat | "
          f"AUC_tr={_fmt_auc(auc_tr0)}  AUC_vl={_fmt_auc(auc_vl0)}")
    tabla.append({
        "paso": 0, "metodo": "Estado_Inicial", "threshold": "-",
        "n_features": len(features),
        "auc_train": auc_tr0, "auc_val": auc_vl0,
        "acc_train": metricas0["acc_train"],
        "acc_val": metricas0["acc_val"],
        "rec_train": metricas0["rec_train"],
        "rec_val": metricas0["rec_val"],
        "prec_train": metricas0["prec_train"],
        "prec_val": metricas0["prec_val"],
        "f1_train": metricas0["f1_train"],
        "f1_val": metricas0["f1_val"],
        "gini_train": metricas0["gini_train"],
        "gini_val": metricas0["gini_val"],
        "n_eliminadas": 0, "features_eliminadas": [], "elegido": True,
    })
    auc_vl_base = auc_vl0   # referencia que se actualiza con cada paso

    # ── Paso 1: missing_variable_method ──────────────────────────────────
    print(f"\n  --- Paso 1: missing_variable_method ---")
    candidatos_p1 = []
    for umbral in UMBRALES_MISSINGS:
        f_ok, elim = filtrar_missings(X_train, features, umbral)
        _, auc_vl = evaluar_rf(X_train, y_train, X_val, y_val, f_ok)
        candidatos_p1.append((f_ok, elim, str(umbral), auc_vl))

    ganador1 = elegir_mejor(candidatos_p1, auc_vl_base)
    for f_ok, elim, umb_str, _ in candidatos_p1:
        es_gan = (ganador1 is not None and umb_str == ganador1[2])
        registrar(1, "missing_variable_method", umb_str, f_ok, elim, es_gan)

    if ganador1:
        features = ganador1[0]
        auc_vl_base = ganador1[3]
    else:
        print("         → ningún umbral mejora: se conserva conjunto base")

    # ── Paso 2: PSI_method ────────────────────────────────────────────────
    print(f"\n  --- Paso 2: PSI_method ---")
    f_ok_psi, psi_vals, elim_psi = filtrar_psi(X_train, X_val, features)
    _, auc_vl_psi = evaluar_rf(X_train, y_train, X_val, y_val, f_ok_psi)
    candidatos_p2 = [
        (f_ok_psi, elim_psi, f">{UMBRAL_PSI}_bins={BINS_PSI}", auc_vl_psi)]
    ganador2 = elegir_mejor(candidatos_p2, auc_vl_base)
    es_gan = ganador2 is not None
    registrar(2, "PSI_method", f">{UMBRAL_PSI}_bins={BINS_PSI}",
              f_ok_psi, elim_psi, es_gan)
    if ganador2:
        features = ganador2[0]
        auc_vl_base = ganador2[3]
    else:
        print("         → PSI no mejora: se conserva conjunto base")

    # ── Paso 3: correlation_method ────────────────────────────────────────
    print(f"\n  --- Paso 3: correlation_method ---")
    candidatos_p3 = []
    for umbral in UMBRALES_CORRELACION:
        f_ok, elim = filtrar_correlacion(X_train, features, umbral)
        _, auc_vl = evaluar_rf(X_train, y_train, X_val, y_val, f_ok)
        candidatos_p3.append((f_ok, elim, str(umbral), auc_vl))

    ganador3 = elegir_mejor(candidatos_p3, auc_vl_base)
    for f_ok, elim, umb_str, _ in candidatos_p3:
        es_gan = (ganador3 is not None and umb_str == ganador3[2])
        registrar(3, "correlation_method", umb_str, f_ok, elim, es_gan)

    if ganador3:
        features = ganador3[0]
        auc_vl_base = ganador3[3]
    else:
        print("         → ningún umbral de correlación mejora: se conserva conjunto base")

    # ── Paso 4: univariante_method ────────────────────────────────────────
    # IMPORTANTE: el umbral aquí es el mínimo poder predictivo individual
    # que debe tener una feature para MANTENERSE.
    # Con umbral=0.10 → se necesita |AUC-0.5| >= 0.10 (AUC >= 0.60 o <= 0.40)
    # Con umbral=0.30 → se necesita |AUC-0.5| >= 0.30 (AUC >= 0.80 o <= 0.20) ← muy restrictivo
    # → Se prueban todos en paralelo y se elige el que maximiza AUC_val
    print(f"\n  --- Paso 4: univariante_method ---")
    candidatos_p4 = []
    for umbral in UMBRALES_UNIVARIANTE:
        f_ok, aucs_uni, elim = filtrar_univariante(
            X_train, y_train, X_val, y_val, features, umbral
        )
        _, auc_vl = evaluar_rf(X_train, y_train, X_val, y_val, f_ok)
        candidatos_p4.append((f_ok, elim, str(umbral), auc_vl))

    ganador4 = elegir_mejor(candidatos_p4, auc_vl_base)
    for f_ok, elim, umb_str, _ in candidatos_p4:
        es_gan = (ganador4 is not None and umb_str == ganador4[2])
        registrar(4, "univariante_method", umb_str, f_ok, elim, es_gan)

    if ganador4:
        features = ganador4[0]
        auc_vl_base = ganador4[3]
    else:
        print("         → ningún umbral univariante mejora: se conserva conjunto base")

    # ── Paso 5 : variance_threshold_method ───────────────────────
    print(f"\n  --- Paso 5 : variance_threshold_method ---")
    f_ok_var, elim_var = filtrar_varianza(X_train, features, UMBRAL_VARIANZA)
    _, auc_vl_var = evaluar_rf(X_train, y_train, X_val, y_val, f_ok_var)
    candidatos_p5 = [(f_ok_var, elim_var, str(UMBRAL_VARIANZA), auc_vl_var)]
    ganador5 = elegir_mejor(candidatos_p5, auc_vl_base)
    es_gan = ganador5 is not None
    registrar(5, "variance_threshold_method ",
              str(UMBRAL_VARIANZA), f_ok_var, elim_var, es_gan)
    if ganador5:
        features = ganador5[0]
        auc_vl_base = ganador5[3]
    else:
        print("         → Variance Threshold no mejora: se conserva conjunto base")

    # ── Paso 6 [EXTRA 2]: rf_importance_method ────────────────────────────
    print(f"\n  --- Paso 6 : rf_importance_method ---")
    candidatos_p6 = []
    for top_n in RF_IMPORTANCE_TOP_NS:
        f_ok, importancias, elim = filtrar_rf_importance(
            X_train, y_train, features, top_n
        )
        _, auc_vl = evaluar_rf(X_train, y_train, X_val, y_val, f_ok)
        candidatos_p6.append((f_ok, elim, f"top_{top_n}", auc_vl))

    ganador6 = elegir_mejor(candidatos_p6, auc_vl_base)
    for f_ok, elim, umb_str, _ in candidatos_p6:
        es_gan = (ganador6 is not None and umb_str == ganador6[2])
        registrar(
            6, "rf_importance_method [EXTRA]", umb_str, f_ok, elim, es_gan)

    if ganador6:
        features = ganador6[0]
        auc_vl_base = ganador6[3]
    else:
        print("         → RF Importance no mejora: se conserva conjunto base")

    return features, tabla


# ---------------------------------------------------------------------------
# 7. GUARDAR
# ---------------------------------------------------------------------------


def guardar_splits(
    X_train, y_train,
    X_val, y_val,
    X_backtest, y_backtest,
    X_live, y_live,
    features_finales
) -> None:
    os.makedirs(MASTER_PATH, exist_ok=True)

    splits = {
        "X_train": X_train[features_finales],
        "y_train": y_train.to_frame(),
        "X_val": X_val[features_finales],
        "y_val": y_val.to_frame(),
        "X_backtest": X_backtest[features_finales],
        "y_backtest": y_backtest.to_frame(),
        "X_live": X_live[features_finales],
        "y_live": y_live.to_frame(),
    }
    for nombre, datos in splits.items():
        ruta_csv  = os.path.join(MASTER_PATH, f"{nombre}.csv")
        ruta_xlsx = os.path.join(MASTER_PATH, f"{nombre}.xlsx")
        shape_str = str(datos.shape) if hasattr(datos, "shape") else ""
        intentos = 0
        while True:
            try:
                datos.to_csv(ruta_csv, index=False, encoding="utf-8")
                print(f"  Guardado: {nombre}.csv  {shape_str}")
                break
            except PermissionError:
                intentos += 1
                if intentos == 1:
                    print(f"\n  [AVISO] '{nombre}.csv' esta abierto en otro programa (Excel?).")
                    print(f"          Cierra el archivo y presiona ENTER para reintentar...")
                    input()
                elif intentos > 3:
                    raise PermissionError(
                        f"No se pudo guardar '{ruta_csv}' tras 3 intentos. "
                        f"Cierra el archivo manualmente y vuelve a ejecutar."
                    )
                else:
                    print(f"          Reintentando ({intentos}/3)... presiona ENTER.")
                    input()
        # Guardar también en XLSX para visualización
        intentos_x = 0
        while True:
            try:
                df_xlsx = datos.reset_index(drop=True) if isinstance(datos, pd.Series) else datos
                if isinstance(datos, pd.Series):
                    df_xlsx = datos.to_frame()
                df_xlsx.to_excel(ruta_xlsx, index=False, engine="openpyxl")
                print(f"  Guardado: {nombre}.xlsx  {shape_str}")
                break
            except PermissionError:
                intentos_x += 1
                if intentos_x == 1:
                    print(f"\n  [AVISO] '{nombre}.xlsx' esta abierto en Excel. Cierra y presiona ENTER...")
                    input()
                elif intentos_x > 3:
                    print(f"  [AVISO] No se pudo guardar '{nombre}.xlsx'. Continúa con CSV.")
                    break
                else:
                    print(f"          Reintentando ({intentos_x}/3)... presiona ENTER.")
                    input()
            except Exception as e:
                print(f"  [AVISO] No se pudo guardar '{nombre}.xlsx': {e}")
                break


def guardar_reportes(features_finales, tabla, estado) -> None:
    os.makedirs(REPORTES_PATH, exist_ok=True)

    # features_seleccionadas.json
    with open(os.path.join(REPORTES_PATH, "features_seleccionadas.json"),
              "w", encoding="utf-8") as f:
        json.dump({"n_features": len(features_finales),
                   "features": features_finales}, f, indent=2)

    # tabla_seleccion_variables.csv (sin la lista de features)
    tabla_csv = [{k: v for k, v in fila.items()
                  if k != "features_eliminadas"}
                 for fila in tabla]
    pd.DataFrame(tabla_csv).to_csv(
        os.path.join(REPORTES_PATH, "tabla_seleccion_variables.csv"), index=False
    )

    # tabla_detalle_eliminadas.csv
    detalle = [
        {"paso": f["paso"], "metodo": f["metodo"],
         "threshold": f["threshold"], "feature_eliminada": feat,
         "paso_elegido": f["elegido"]}
        for f in tabla
        for feat in f.get("features_eliminadas", [])
    ]
    pd.DataFrame(detalle).to_csv(
        os.path.join(REPORTES_PATH, "tabla_detalle_eliminadas.csv"), index=False
    )

    # estado_seleccion.json
    with open(os.path.join(REPORTES_PATH, "estado_seleccion.json"),
              "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, ensure_ascii=False, default=str)

    print(
        f"  features_seleccionadas.json     ({len(features_finales)} features)")
    print(f"  tabla_seleccion_variables.csv")
    print(f"  tabla_detalle_eliminadas.csv")
    print(f"  estado_seleccion.json")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _assert_sin_nulos(paso: str, **dfs):
    """Aborta el script si cualquier DataFrame tiene nulos."""
    for nombre, df in dfs.items():
        n = int(df.isnull().sum().sum())
        if n > 0:
            cols = df.columns[df.isnull().any()].tolist()
            raise ValueError(
                f"\n[ERROR NULOS] Paso '{paso}' — '{nombre}' tiene {n} nulos "
                f"en columnas: {cols}\n"
                f"Verifica que master_table_limpia.csv este actualizado "
                f"(ejecuta 03_limpieza.py primero)."
            )


def main():
    inicio = datetime.now()
    print("=" * 80)
    print("SCRIPT 04 — SPLIT + TRANSFORMACIONES + SELECCIÓN DE VARIABLES")
    print(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\n[1] Cargando master table limpia …")
    df = cargar_master()
    _assert_sin_nulos("carga", master=df)

    print("\n[2] Split temporal …")
    train, val, backtest, live = split_temporal(df)
    _assert_sin_nulos("split", train=train, val=val, backtest=backtest, live=live)

    print("\n[3] Separando X / y …")
    X_train,    y_train    = separar_Xy(train)
    X_val,      y_val      = separar_Xy(val)
    X_backtest, y_backtest = separar_Xy(backtest)
    X_live,     y_live     = separar_Xy(live)
    print(f"  X_train shape: {X_train.shape}")
    _assert_sin_nulos("sep_Xy",
                      X_train=X_train, X_val=X_val,
                      X_backtest=X_backtest, X_live=X_live)

    print("\n[4] Target Encoding (fit en train) …")
    X_train, X_val, X_backtest, X_live, te_maps = aplicar_target_encoding(
        X_train, y_train, X_val, X_backtest, X_live
    )
    _assert_sin_nulos("target_encoding",
                      X_train=X_train, X_val=X_val,
                      X_backtest=X_backtest, X_live=X_live)

    print("\n[5] StandardScaler (fit en train) …")
    X_train, X_val, X_backtest, X_live, scaler, cols_num = aplicar_scaler(
        X_train, X_val, X_backtest, X_live
    )
    _assert_sin_nulos("scaler",
                      X_train=X_train, X_val=X_val,
                      X_backtest=X_backtest, X_live=X_live)

    print("\n[6] Eliminando columnas object residuales …")
    cols_object = X_train.select_dtypes(
        include=["object", "category"]).columns.tolist()
    if cols_object:
        print(f"  Columnas object eliminadas: {cols_object}")
        for df_part in [X_train, X_val, X_backtest, X_live]:
            df_part.drop(columns=cols_object, inplace=True, errors="ignore")
    else:
        print("  Sin columnas object residuales.")
    _assert_sin_nulos("drop_object",
                      X_train=X_train, X_val=X_val,
                      X_backtest=X_backtest, X_live=X_live)

    print("\n[7] Selección de variables …")
    print("=" * 80)
    features_finales, tabla = seleccion_de_variables(
        X_train, y_train, X_val, y_val
    )
    print("=" * 80)

    print("\n[8] Guardando splits …")
    guardar_splits(
        X_train, y_train,
        X_val, y_val,
        X_backtest, y_backtest,
        X_live, y_live,
        features_finales
    )

    print("\n[9] Guardando reportes …")
    estado = {
        "script": "04_seleccion_variables",
        "timestamp": datetime.now().isoformat(),
        "duracion_seg": round((datetime.now() - inicio).total_seconds(), 2),
        "tolerancia_auc": TOLERANCIA_AUC,
        "cortes_temporales": {
            "train_hasta": FECHA_FIN_TRAIN,
            "val_hasta": FECHA_FIN_VAL,
            "backtest_hasta": FECHA_FIN_BACKTEST,
        },
        "shapes": {
            "train": {"filas": len(X_train),    "features": len(features_finales)},
            "val": {"filas": len(X_val),      "features": len(features_finales)},
            "backtest": {"filas": len(X_backtest), "features": len(features_finales)},
            "live": {"filas": len(X_live),     "features": len(features_finales)},
        },
        "features_finales": features_finales,
        "tabla_seleccion": tabla,
        "te_maps": {k: len(v) for k, v in te_maps.items()},
        "parametros": {
            "umbrales_missings": UMBRALES_MISSINGS,
            "umbral_psi": UMBRAL_PSI,
            "bins_psi": BINS_PSI,
            "umbrales_correlacion": UMBRALES_CORRELACION,
            "umbrales_univariante": UMBRALES_UNIVARIANTE,
            "umbral_varianza": UMBRAL_VARIANZA,
            "rf_importance_top_ns": RF_IMPORTANCE_TOP_NS,
        },
    }
    guardar_reportes(features_finales, tabla, estado)

    # ── Tabla resumen final ───────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("TABLA RESUMEN — SELECCIÓN DE VARIABLES")
    print(
        f"  {'P':<2} {'Método':<38} {'Threshold':<16} {'Feat':>5} "
        f"{'AUC Tr':>8} {'AUC Val':>8} {'Acc Tr':>8} {'Acc Val':>8} "
        f"{'Rec Tr':>8} {'Rec Val':>8} {'Prec Tr':>8} {'Prec Val':>8} "
        f"{'F1 Tr':>8} {'F1 Val':>8} {'Gini Tr':>8} {'Gini Val':>8} "
        f"{'Elim':>5} {'':>10}"
    )
    print("-" * 120)

    for row in tabla:
        marca = " ◄" if row.get("elegido") else ""
        auc_tr = _fmt_auc(row["auc_train"])
        auc_vl = _fmt_auc(row["auc_val"])

        # Formatear métricas adicionales
        acc_tr = _fmt_auc(row.get("acc_train"))
        acc_vl = _fmt_auc(row.get("acc_val"))
        rec_tr = _fmt_auc(row.get("rec_train"))
        rec_vl = _fmt_auc(row.get("rec_val"))
        prec_tr = _fmt_auc(row.get("prec_train"))
        prec_vl = _fmt_auc(row.get("prec_val"))
        f1_tr = _fmt_auc(row.get("f1_train"))
        f1_vl = _fmt_auc(row.get("f1_val"))
        gini_tr = _fmt_auc(row.get("gini_train"))
        gini_vl = _fmt_auc(row.get("gini_val"))

        print(
            f"  {row['paso']:<2} {row['metodo']:<38} {str(row['threshold']):<16} "
            f"{row['n_features']:>5} {auc_tr:>8} {auc_vl:>8} "
            f"{acc_tr:>8} {acc_vl:>8} "
            f"{rec_tr:>8} {rec_vl:>8} "
            f"{prec_tr:>8} {prec_vl:>8} "
            f"{f1_tr:>8} {f1_vl:>8} "
            f"{gini_tr:>8} {gini_vl:>8} "
            f"{row['n_eliminadas']:>5}{marca}"
        )
    print("=" * 120)
    print(f"\nFeatures finales seleccionadas : {len(features_finales)}")
    print(f"Features finales               : {features_finales}")
    print(f"Duración total                 : {estado['duracion_seg']}s")

    # ── Verificación final de integridad ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERIFICACION FINAL DE INTEGRIDAD")
    print("=" * 60)

    splits_guardados = {
        "X_train":    (os.path.join(MASTER_PATH, "X_train.csv"),    len(X_train)),
        "y_train":    (os.path.join(MASTER_PATH, "y_train.csv"),    len(y_train)),
        "X_val":      (os.path.join(MASTER_PATH, "X_val.csv"),      len(X_val)),
        "y_val":      (os.path.join(MASTER_PATH, "y_val.csv"),      len(y_val)),
        "X_backtest": (os.path.join(MASTER_PATH, "X_backtest.csv"), len(X_backtest)),
        "y_backtest": (os.path.join(MASTER_PATH, "y_backtest.csv"), len(y_backtest)),
        "X_live":     (os.path.join(MASTER_PATH, "X_live.csv"),     len(X_live)),
        "y_live":     (os.path.join(MASTER_PATH, "y_live.csv"),     len(y_live)),
    }

    total_filas = 0
    errores = []
    print(f"  {'Archivo':<22} {'Filas':>8}  {'Nulos':>6}  {'Cols':>5}  Estado")
    print("  " + "-" * 55)

    for nombre, (ruta, filas_mem) in splits_guardados.items():
        df_check = pd.read_csv(ruta, low_memory=False)
        nulos    = int(df_check.isnull().sum().sum())
        cols     = df_check.shape[1]
        ok_filas = df_check.shape[0] == filas_mem
        ok_nulos = nulos == 0

        estado_str = "OK" if (ok_filas and ok_nulos) else ""
        if not ok_filas:
            estado_str += f" [!] filas disco={df_check.shape[0]} vs memoria={filas_mem}"
            errores.append(nombre)
        if not ok_nulos:
            cols_null = df_check.columns[df_check.isnull().any()].tolist()
            estado_str += f" [!] {nulos} nulos en {cols_null}"
            errores.append(nombre)

        print(f"  {nombre:<22} {df_check.shape[0]:>8,}  {nulos:>6}  {cols:>5}  {estado_str}")

        if nombre.startswith("X_"):
            total_filas += df_check.shape[0]

    print("  " + "-" * 55)
    print(f"  {'Total filas X_*':<22} {total_filas:>8,}")
    print()

    if errores:
        print(f"  [ATENCION] Problemas detectados en: {errores}")
    else:
        print("  Todos los splits son correctos: sin nulos, filas cuadran con master.")

    print("=" * 60)
    print("Script 04 completado.\n")


if __name__ == "__main__":
    main()
