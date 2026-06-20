"""
=============================================================================
Script 09 — Análisis SHAP sobre X_live   (Logistic Regression)
Proyecto : Predicción de Satisfacción del Cliente — Olist
Grupo 1  | Sprint 4
=============================================================================
Responsabilidad:
  1. Verificar que exista el PKL del modelo (Logistic Regression).
  2. Cargar X_live / y_live.
  3. Calcular valores SHAP con LinearExplainer (fit para modelos lineales).
  4. Generar y guardar:
       shap_live_01_bar.png          — Importancia global (mean |SHAP|)
       shap_live_02_beeswarm.png     — Importancia + dirección del efecto
       shap_live_03_waterfall.png    — Explicación local (primera obs.)
       shap_live_04_dep_<feat>.png   — Dependence plots de top-5 features
       shap_live_coeficientes.csv    — Coeficientes del modelo
  5. Guardar estado_shap_live.json con resumen de ejecución.

Entradas : data/reportes/modelos_optuna_cls0.pkl  (o modelos_optuna.pkl)
           data/master/X_live.csv
           data/master/y_live.csv
Salidas  : reports/sprint4/shap_live_*.png
           data/reportes/estado_shap_live.json
           data/reportes/shap_live_coeficientes.csv
=============================================================================
"""

from __future__ import annotations

import json
import sys
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
import shap

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
REPORTES = ROOT / "data" / "reportes"
DATA     = ROOT / "data" / "master"
REPORTS  = ROOT / "reports" / "sprint4"
REPORTS.mkdir(parents=True, exist_ok=True)

MODELO_ELEGIDO = "Logistic Regression"

# PKL candidates — el mismo orden que usa 08_backtest.py
PKL_CANDIDATES = [
    REPORTES / "modelos_optuna_cls0.pkl",
    REPORTES / "modelos_optuna.pkl",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, nombre: str) -> str:
    path = REPORTS / nombre
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> reports/sprint4/{nombre}")
    return nombre


# ---------------------------------------------------------------------------
# 1. Verificación y carga del PKL
# ---------------------------------------------------------------------------

def verificar_y_cargar_modelo():
    """Busca el PKL en la lista de candidatos. Aborta si no existe ninguno."""
    pkl_encontrado = None
    for candidate in PKL_CANDIDATES:
        if candidate.exists():
            pkl_encontrado = candidate
            break

    if pkl_encontrado is None:
        rutas_str = "\n  ".join(str(p) for p in PKL_CANDIDATES)
        raise FileNotFoundError(
            f"No se encontró el archivo PKL del modelo.\n"
            f"Rutas buscadas:\n  {rutas_str}\n"
            f"Ejecuta primero 06_optuna.py para generar el PKL."
        )

    print(f"  PKL encontrado : {pkl_encontrado.name}")
    todos = joblib.load(pkl_encontrado)

    if MODELO_ELEGIDO not in todos:
        raise KeyError(
            f"El modelo '{MODELO_ELEGIDO}' no está en el PKL.\n"
            f"Modelos disponibles: {list(todos.keys())}"
        )

    modelo = todos[MODELO_ELEGIDO]
    print(f"  Modelo cargado : {MODELO_ELEGIDO}")
    print(f"  Tipo           : {type(modelo).__name__}")
    return modelo, str(pkl_encontrado.name)


# ---------------------------------------------------------------------------
# 2. Carga de X_live / y_live
# ---------------------------------------------------------------------------

def cargar_live():
    ruta_x = DATA / "X_live.csv"
    ruta_y = DATA / "y_live.csv"

    for ruta in [ruta_x, ruta_y]:
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró {ruta}.\n"
                f"Ejecuta primero 04_split.py para generar los splits."
            )

    X = pd.read_csv(ruta_x)
    y_df = pd.read_csv(ruta_y)
    y = y_df["satisfecho"] if "satisfecho" in y_df.columns else y_df.squeeze()

    print(f"  X_live shape   : {X.shape}")
    print(f"  y_live         : {len(y)} filas  "
          f"| satisfechos={y.mean():.2%}  insatisfechos={(1-y.mean()):.2%}")
    return X, y


# ---------------------------------------------------------------------------
# 3. SHAP — LinearExplainer
# ---------------------------------------------------------------------------

def calcular_shap(modelo, X: pd.DataFrame):
    print(f"  Calculando SHAP values con LinearExplainer sobre {len(X):,} filas...")
    explainer  = shap.LinearExplainer(modelo, X)
    shap_vals  = explainer(X)
    print(f"  shap_values shape: {shap_vals.shape}")
    return explainer, shap_vals


# ---------------------------------------------------------------------------
# 4. Gráficos
# ---------------------------------------------------------------------------

def plot_bar(shap_vals, nombre="shap_live_01_bar.png") -> str:
    """Importancia global — media de |SHAP| por feature."""
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.bar(shap_vals, show=False, ax=ax)
    ax.set_title(
        f"Importancia Global — {MODELO_ELEGIDO}\n(Live, mean |SHAP value|)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, nombre)


def plot_beeswarm(shap_vals, nombre="shap_live_02_beeswarm.png") -> str:
    """Beeswarm/Summary — importancia + dirección del efecto."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_vals, show=False)
    plt.title(
        f"Importancia y Dirección del Efecto — {MODELO_ELEGIDO}\n(Live)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, nombre)


def plot_waterfall(shap_vals, idx: int = 0,
                   nombre="shap_live_03_waterfall.png") -> str:
    """Explicación local de una observación individual."""
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_vals[idx], show=False)
    plt.title(
        f"Explicación Local — Observación #{idx}  ({MODELO_ELEGIDO} | Live)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return _save(fig, nombre)


def plot_dependencias(shap_vals, X: pd.DataFrame,
                      top_n: int = 5) -> list[str]:
    """Dependence plots para las top-N features más importantes."""
    importancia = np.abs(shap_vals.values).mean(axis=0)
    top_idx     = np.argsort(importancia)[::-1][:top_n]

    print(f"\n  Top {top_n} features más importantes:")
    for rank, i in enumerate(top_idx, 1):
        print(f"    {rank}. {X.columns[i]}: {importancia[i]:.4f}")

    plt.style.use("bmh")
    generados = []
    for rank, i in enumerate(top_idx, 1):
        nombre_feat = X.columns[i]
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.scatter(shap_vals[:, nombre_feat], show=False, cmap="coolwarm")
        plt.title(
            f"Dependencia SHAP: {nombre_feat}  (Live | rank #{rank})",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        archivo = f"shap_live_04_dep_{nombre_feat}.png"
        generados.append(_save(fig, archivo))

    plt.style.use("default")
    return generados


# ---------------------------------------------------------------------------
# 5. Exportar coeficientes del modelo
# ---------------------------------------------------------------------------

def guardar_coeficientes(modelo, X: pd.DataFrame) -> str:
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coef":    modelo.coef_[0],
    }).sort_values("coef", ascending=False).reset_index(drop=True)

    ruta = REPORTES / "shap_live_coeficientes.csv"
    coef_df.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"   -> data/reportes/shap_live_coeficientes.csv")

    print("\n  Coeficientes del modelo (ordenados desc):")
    for _, row in coef_df.iterrows():
        signo = "+" if row["coef"] >= 0 else ""
        print(f"    {row['feature']:<40} {signo}{row['coef']:.6f}")

    return str(ruta)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t_inicio = time.time()
    print("=" * 65)
    print("09_shap_live.py — Análisis SHAP sobre X_live")
    print(f"Modelo: {MODELO_ELEGIDO}")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # [1] Verificación y carga del modelo
    print("\n[1] Verificando PKL y cargando modelo...")
    modelo, pkl_usado = verificar_y_cargar_modelo()

    # [2] Carga de datos live
    print("\n[2] Cargando X_live / y_live...")
    X_live, y_live = cargar_live()

    # [3] Cálculo de SHAP
    print("\n[3] Calculando SHAP values...")
    explainer, shap_vals = calcular_shap(modelo, X_live)

    # [4] Gráficos
    print("\n[4] Generando gráficos SHAP...\n")
    generados = []

    generados.append(plot_bar(shap_vals))
    generados.append(plot_beeswarm(shap_vals))
    generados.append(plot_waterfall(shap_vals, idx=0))
    generados.extend(plot_dependencias(shap_vals, X_live, top_n=5))

    # [5] Coeficientes
    print("\n[5] Exportando coeficientes del modelo...")
    guardar_coeficientes(modelo, X_live)

    # [6] Estado JSON
    duracion = round(time.time() - t_inicio, 2)
    importancia = np.abs(shap_vals.values).mean(axis=0)
    top5_idx    = np.argsort(importancia)[::-1][:5]

    estado = {
        "script":           "09_shap_live",
        "timestamp":        datetime.now().isoformat(),
        "duracion_seg":     duracion,
        "modelo_elegido":   MODELO_ELEGIDO,
        "pkl_usado":        pkl_usado,
        "datos": {
            "X_live_shape":  list(X_live.shape),
            "n_filas":       len(X_live),
            "n_features":    X_live.shape[1],
            "tasa_satisfecho": round(float(y_live.mean()), 4),
        },
        "shap": {
            "explainer_tipo": "LinearExplainer",
            "shap_values_shape": list(shap_vals.shape),
            "top5_features": [
                {
                    "rank":     int(rank + 1),
                    "feature":  str(X_live.columns[i]),
                    "mean_abs_shap": round(float(importancia[i]), 6),
                }
                for rank, i in enumerate(top5_idx)
            ],
        },
        "graficos_generados": generados,
        "directorio_salida":  "reports/sprint4",
    }

    ruta_estado = REPORTES / "estado_shap_live.json"
    with open(ruta_estado, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, ensure_ascii=False)

    # Resumen final
    print(f"\n{'='*65}")
    print(f"[OK] Análisis SHAP completado en {duracion}s")
    print(f"     {len(generados)} gráficos generados en reports/sprint4/")
    print(f"     data/reportes/estado_shap_live.json")
    print(f"     data/reportes/shap_live_coeficientes.csv")
    print(f"\nTop 5 features (mean |SHAP|):")
    for entry in estado["shap"]["top5_features"]:
        print(f"  {entry['rank']}. {entry['feature']:<40} {entry['mean_abs_shap']:.6f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
