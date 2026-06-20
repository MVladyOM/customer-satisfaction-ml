"""
=============================================================================
Script 03 - Limpieza de la Master Table
Proyecto : Predicción de Satisfacción del Cliente — Olist
Grupo 1  | Sprint 2
=============================================================================
Responsabilidad:
  1. Eliminar columnas no modelables:
       - IDs (no tienen poder predictivo)
       - Fechas crudas (ya se extrajeron todas las features derivadas)
       - Columnas derivadas del target (leakage directo)
  2. Eliminar columnas con missings > umbral (15%)
  3. Imputar missings restantes:
       - Numéricas  → mediana  (robusta ante outliers)
       - Categóricas → 'desconocido' (preserva la ausencia como señal)
  4. Agrupar categorías raras (< 1% frecuencia) → 'otros'
  5. Verificar coherencia final: shape, nulos residuales, tipos

IMPORTANTE — Escalado y codificación NO se hacen aquí.
  StandardScaler y Target Encoding se fitean SOLO sobre X_train en
  el Script 04, después del split temporal, para evitar data leakage.

Entrada : data/master/master_table.csv
Salida  : data/master/master_table_limpia.csv
          data/reportes/estado_limpieza.json
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
MASTER_PATH   = os.path.join("data", "master")
REPORTES_PATH = os.path.join("data", "reportes")

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
UMBRAL_MISSINGS    = 0.15   # eliminar columna si tiene > 15% de nulos
UMBRAL_CATEGORIA   = 0.01   # agrupar categoría si frecuencia < 1%

# ---------------------------------------------------------------------------
# Columnas a eliminar explícitamente
# ---------------------------------------------------------------------------

# IDs: no tienen poder predictivo directo
# order_id se conserva como identificador para trazabilidad en y_*
COLS_IDS = [
    "customer_id",
    "seller_id",
    "product_id",
]

# Fechas crudas: todas sus features derivadas ya están en la master table
COLS_FECHAS = [
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "purchase_year_month",   # periodo en string, se usó para lag features
]

# Leakage directo del target: son consecuencia de la insatisfacción,
# no causas. Incluirlas equivale a "mirar el futuro".
#   - review_score     → el target sin binarizar
#   - review_comment_*  → el cliente escribe el comentario DESPUÉS
#                         de haber decidido su score
#   - has_comment       → derivada de review_comment_message
#   - has_review_title  → derivada de review_comment_title
#   - comment_length    → derivada de review_comment_message
COLS_LEAKAGE = [
    "review_score",
    "review_comment_message",
    "review_comment_title",
    "has_comment",
    "has_review_title",
    "comment_length",
]

# Columnas duplicadas o redundantes
COLS_DUPLICADAS = [
    "uses_installments",    # duplicado de pago_en_cuotas
    "customer_city",        # 4119 valores únicos — redundante, ya tienes customer_state
    "seller_city",          # 611 valores únicos  — redundante, ya tienes seller_state
]

COLS_ELIMINAR = COLS_IDS + COLS_FECHAS + COLS_LEAKAGE + COLS_DUPLICADAS

# ---------------------------------------------------------------------------
# Columnas categóricas con alta cardinalidad para agrupar raras
# ---------------------------------------------------------------------------
COLS_CATEGORICAS = [
    "product_category_name_english",
    "customer_state",
    "seller_state",
    "payment_type",
    "customer_region",
]


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_master() -> pd.DataFrame:
    ruta = os.path.join(MASTER_PATH, "master_table.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró {ruta}. Ejecuta primero el Script 02."
        )
    df = pd.read_csv(ruta, low_memory=False)
    print(f"  Master Table cargada: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def eliminar_columnas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Elimina IDs, fechas crudas, leakage y duplicados."""
    presentes  = [c for c in COLS_ELIMINAR if c in df.columns]
    ausentes   = [c for c in COLS_ELIMINAR if c not in df.columns]
    df = df.drop(columns=presentes)

    log = {
        "eliminadas_total"  : len(presentes),
        "eliminadas_ids"    : [c for c in COLS_IDS       if c in presentes],
        "eliminadas_fechas" : [c for c in COLS_FECHAS    if c in presentes],
        "eliminadas_leakage": [c for c in COLS_LEAKAGE   if c in presentes],
        "eliminadas_duplic" : [c for c in COLS_DUPLICADAS if c in presentes],
        "no_encontradas"    : ausentes,
    }

    print(f"  Columnas eliminadas: {len(presentes)}")
    if ausentes:
        print(f"  No encontradas (ya no existen): {ausentes}")
    return df, log


def eliminar_por_missings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Elimina columnas cuyo porcentaje de nulos supera el umbral."""
    pct_nulos = df.isnull().mean()
    cols_drop  = pct_nulos[pct_nulos > UMBRAL_MISSINGS].index.tolist()

    log = {f"{c}": round(float(pct_nulos[c]), 4) for c in cols_drop}

    if cols_drop:
        df = df.drop(columns=cols_drop)
        print(f"  Columnas eliminadas por missings > {UMBRAL_MISSINGS:.0%}: {cols_drop}")
    else:
        print(f"  Ninguna columna supera el umbral de missings ({UMBRAL_MISSINGS:.0%})")

    return df, log


def imputar_missings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Imputa nulos restantes:
      - Numéricas  → mediana (robusta ante la distribución sesgada de Olist)
      - Categóricas → 'desconocido' (preserva la ausencia como categoría)
    """
    log = {}

    numericas    = df.select_dtypes(include=[np.number]).columns.tolist()
    categoricas  = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in numericas:
        n_nulos = df[col].isnull().sum()
        if n_nulos > 0:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            log[col] = {"tipo": "mediana", "valor": round(float(mediana), 4), "nulos_imputados": int(n_nulos)}

    for col in categoricas:
        n_nulos = df[col].isnull().sum()
        if n_nulos > 0:
            df[col] = df[col].fillna("desconocido")
            log[col] = {"tipo": "constante", "valor": "desconocido", "nulos_imputados": int(n_nulos)}

    total_imputado = sum(v["nulos_imputados"] for v in log.values())
    print(f"  Celdas imputadas: {total_imputado:,} en {len(log)} columnas")
    return df, log


def agrupar_categorias_raras(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Para cada columna categórica, reemplaza categorías con frecuencia
    relativa < UMBRAL_CATEGORIA por 'otros'.
    Reduce cardinalidad y evita que el modelo aprenda de grupos
    con demasiado poco soporte estadístico.
    """
    log = {}

    for col in COLS_CATEGORICAS:
        if col not in df.columns:
            continue

        frecuencias  = df[col].value_counts(normalize=True)
        cats_raras   = frecuencias[frecuencias < UMBRAL_CATEGORIA].index.tolist()

        if cats_raras:
            df[col] = df[col].where(~df[col].isin(cats_raras), other="otros")
            log[col] = {
                "categorias_agrupadas" : len(cats_raras),
                "ejemplos"             : cats_raras[:5],
                "cardinalidad_final"   : int(df[col].nunique()),
            }
            print(f"  {col}: {len(cats_raras)} categorías raras → 'otros'  "
                  f"(cardinalidad final: {df[col].nunique()})")
        else:
            print(f"  {col}: sin categorías raras")

    return df, log


def verificar_coherencia(df: pd.DataFrame) -> dict:
    """Chequeos finales de coherencia antes de guardar."""
    nulos_residuales = int(df.isnull().sum().sum())
    assert nulos_residuales == 0, (
        f"Quedaron {nulos_residuales} nulos residuales. Revisar imputación."
    )

    assert "satisfecho" in df.columns, "Falta la variable target 'satisfecho'."

    assert df["satisfecho"].isin([0, 1]).all(), (
        "La variable 'satisfecho' contiene valores fuera de {0,1}."
    )

    # Verificar que no queden columnas de leakage
    leakage_residual = [c for c in COLS_LEAKAGE if c in df.columns]
    assert not leakage_residual, (
        f"Columnas de leakage aún presentes: {leakage_residual}"
    )

    dist = df["satisfecho"].value_counts(normalize=True)
    print(f"  Target — Satisfecho: {dist.get(1, 0):.1%}  |  "
          f"Insatisfecho: {dist.get(0, 0):.1%}")
    print(f"  Nulos residuales: {nulos_residuales}")
    print(f"  Shape final: {df.shape[0]:,} × {df.shape[1]}")

    return {
        "nulos_residuales" : nulos_residuales,
        "shape_final"      : {"filas": int(df.shape[0]), "columnas": int(df.shape[1])},
        "pct_satisfecho"   : round(float(dist.get(1, 0)), 4),
        "columnas_finales" : list(df.columns),
        "tipos_datos"      : {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def guardar_resultados(df: pd.DataFrame, estado: dict) -> None:
    os.makedirs(MASTER_PATH,   exist_ok=True)
    os.makedirs(REPORTES_PATH, exist_ok=True)

    ruta_csv  = os.path.join(MASTER_PATH,   "master_table_limpia.csv")
    ruta_json = os.path.join(REPORTES_PATH, "estado_limpieza.json")

    df.to_csv(ruta_csv, index=False)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Guardado: {ruta_csv}")
    print(f"  Reporte : {ruta_json}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    inicio = datetime.now()
    print("=" * 65)
    print("SCRIPT 03 — LIMPIEZA DE LA MASTER TABLE")
    print(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    df = cargar_master()
    shape_original = df.shape
    estado = {
        "script"         : "03_limpieza",
        "timestamp"      : datetime.now().isoformat(),
        "shape_original" : {"filas": shape_original[0], "columnas": shape_original[1]},
    }

    print("\n[1] Eliminando columnas no modelables …")
    df, log_cols = eliminar_columnas(df)
    estado["paso_1_eliminacion"] = log_cols

    print("\n[2] Eliminando columnas con missings excesivos …")
    df, log_miss = eliminar_por_missings(df)
    estado["paso_2_missings_altos"] = log_miss

    print("\n[3] Imputando missings restantes …")
    df, log_imput = imputar_missings(df)
    estado["paso_3_imputacion"] = log_imput

    print("\n[4] Agrupando categorías raras …")
    df, log_cats = agrupar_categorias_raras(df)
    estado["paso_4_categorias_raras"] = log_cats

    print("\n[5] Verificando coherencia …")
    log_verif = verificar_coherencia(df)
    estado["paso_5_verificacion"] = log_verif
    estado["duracion_seg"] = round((datetime.now() - inicio).total_seconds(), 2)

    print("\n[6] Guardando …")
    guardar_resultados(df, estado)

    print("\n" + "=" * 65)
    columnas_reducidas = shape_original[1] - df.shape[1]
    print(f"  Columnas originales : {shape_original[1]}")
    print(f"  Columnas eliminadas : {columnas_reducidas}")
    print(f"  Columnas finales    : {df.shape[1]}")
    print(f"  Filas finales       : {df.shape[0]:,}")
    print(f"  Duración            : {estado['duracion_seg']}s")
    print("=" * 65)
    print("Script 03 completado.\n")

    return df


if __name__ == "__main__":
    main()