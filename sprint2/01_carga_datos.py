"""
=============================================================================
Script 01 - Carga de Datos
Proyecto : Predicción de Satisfacción del Cliente — Olist
Grupo 1  | Sprint 2
=============================================================================
Responsabilidad:
  - Leer las 9 tablas CSV desde data/raw
  - Verificar integridad por tabla (shape, nulos, duplicados)
  - Guardar cada tabla en data/processed sin transformar
  - Emitir estado_carga.json con resumen de calidad inicial

Salida:
  data/processed/<tabla>.csv   (9 archivos)
  data/reportes/estado_carga.json
=============================================================================
"""

import os
import json
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAW_PATH       = os.path.join("data", "raw")
PROCESSED_PATH = os.path.join("data", "processed")
REPORTES_PATH  = os.path.join("data", "reportes")

# ---------------------------------------------------------------------------
# Catálogo de tablas
# ---------------------------------------------------------------------------
TABLAS = {
    "orders"      : "olist_orders_dataset.csv",
    "reviews"     : "olist_order_reviews_dataset.csv",
    "items"       : "olist_order_items_dataset.csv",
    "payments"    : "olist_order_payments_dataset.csv",
    "customers"   : "olist_customers_dataset.csv",
    "products"    : "olist_products_dataset.csv",
    "sellers"     : "olist_sellers_dataset.csv",
    "geolocation" : "olist_geolocation_dataset.csv",
    "translation" : "product_category_name_translation.csv",
}

# Columnas de fecha por tabla (para parseo automático)
COLUMNAS_FECHA = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
    "items"  : ["shipping_limit_date"],
}


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------

def cargar_tabla(nombre: str, archivo: str) -> pd.DataFrame | None:
    """Carga un CSV desde RAW_PATH con parseo de fechas si aplica."""
    ruta = os.path.join(RAW_PATH, archivo)

    if not os.path.exists(ruta):
        print(f"  [ERROR] No encontrado: {ruta}")
        return None

    fechas = COLUMNAS_FECHA.get(nombre, [])
    try:
        df = pd.read_csv(ruta, parse_dates=fechas, low_memory=False)
        print(f"  [OK] {nombre:15s} — {df.shape[0]:>7,} filas  ×  {df.shape[1]:>2} columnas")
        return df
    except Exception as e:
        print(f"  [ERROR] {nombre}: {e}")
        return None


def verificar_calidad(nombre: str, df: pd.DataFrame) -> dict:
    """Devuelve métricas de calidad para una tabla."""
    nulos_por_col = df.isnull().sum()
    return {
        "filas"          : int(df.shape[0]),
        "columnas"       : int(df.shape[1]),
        "nulos_total"    : int(nulos_por_col.sum()),
        "columnas_con_nulos": int((nulos_por_col > 0).sum()),
        "duplicados"     : int(df.duplicated().sum()),
        "nulos_detalle"  : {
            col: int(n) for col, n in nulos_por_col.items() if n > 0
        },
    }


def guardar_tabla(nombre: str, df: pd.DataFrame) -> None:
    """Persiste la tabla en data/processed."""
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    ruta = os.path.join(PROCESSED_PATH, f"{nombre}.csv")
    df.to_csv(ruta, index=False)


def guardar_estado(reporte: dict) -> None:
    """Serializa el reporte de calidad como JSON."""
    os.makedirs(REPORTES_PATH, exist_ok=True)
    ruta = os.path.join(REPORTES_PATH, "estado_carga.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Reporte guardado → {ruta}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("SCRIPT 01 — CARGA DE DATOS")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    datos   = {}
    reporte = {
        "script"    : "01_carga_datos",
        "timestamp" : datetime.now().isoformat(),
        "tablas"    : {},
        "resumen"   : {},
    }

    # 1. Carga
    print("\n[1] Cargando tablas desde data/raw …")
    for nombre, archivo in TABLAS.items():
        df = cargar_tabla(nombre, archivo)
        if df is not None:
            datos[nombre] = df

    # 2. Verificación de calidad
    print("\n[2] Verificando calidad por tabla …")
    for nombre, df in datos.items():
        calidad = verificar_calidad(nombre, df)
        reporte["tablas"][nombre] = calidad

        alertas = []
        if calidad["duplicados"] > 0:
            alertas.append(f"{calidad['duplicados']} duplicados")
        if calidad["nulos_total"] > 0:
            alertas.append(f"{calidad['nulos_total']} nulos")

        estado = "⚠ REVISAR" if alertas else "✓ OK"
        print(f"  {nombre:15s} {estado}  {' | '.join(alertas) if alertas else ''}")

    # 3. Guardar en processed
    print("\n[3] Guardando tablas en data/processed …")
    for nombre, df in datos.items():
        guardar_tabla(nombre, df)
        print(f"  Guardado: {nombre}.csv")

    # 4. Resumen global
    total_filas  = sum(v["filas"]       for v in reporte["tablas"].values())
    total_nulos  = sum(v["nulos_total"] for v in reporte["tablas"].values())
    total_dupl   = sum(v["duplicados"]  for v in reporte["tablas"].values())
    tablas_ok    = sum(1 for v in reporte["tablas"].values()
                       if v["nulos_total"] == 0 and v["duplicados"] == 0)

    reporte["resumen"] = {
        "tablas_cargadas"    : len(datos),
        "tablas_esperadas"   : len(TABLAS),
        "tablas_limpias"     : tablas_ok,
        "total_filas"        : total_filas,
        "total_nulos"        : total_nulos,
        "total_duplicados"   : total_dupl,
        "estado_general"     : "OK" if len(datos) == len(TABLAS) else "INCOMPLETO",
    }

    guardar_estado(reporte)

    print("\n" + "=" * 65)
    print("RESUMEN FINAL")
    print(f"  Tablas cargadas : {len(datos)}/{len(TABLAS)}")
    print(f"  Filas totales   : {total_filas:,}")
    print(f"  Nulos totales   : {total_nulos:,}")
    print(f"  Duplicados      : {total_dupl:,}")
    print(f"  Estado          : {reporte['resumen']['estado_general']}")
    print("=" * 65)
    print("Script 01 completado.\n")

    return datos


if __name__ == "__main__":
    main()
