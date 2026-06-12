from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
SPRINT_DIR = Path(__file__).resolve().parent

RAW_ESPERADOS = [
    "data/raw/olist_customers_dataset.csv",
    "data/raw/olist_geolocation_dataset.csv",
    "data/raw/olist_order_items_dataset.csv",
    "data/raw/olist_orders_dataset.csv",
    "data/raw/olist_order_payments_dataset.csv",
    "data/raw/olist_products_dataset.csv",
    "data/raw/olist_order_reviews_dataset.csv",
    "data/raw/olist_sellers_dataset.csv",
    "data/raw/product_category_name_translation.csv",
]

SPRINT2_DIR = ROOT / "sprint2"

PIPELINE = [
    # ── Sprint 2: preparación de datos ──────────────────────────────────────
    {
        "script":  SPRINT2_DIR / "01_carga_datos.py",
        "label":   "01_carga_datos",
        "outputs": [
            "data/processed/customers.csv",
            "data/processed/geolocation.csv",
            "data/processed/items.csv",
            "data/processed/orders.csv",
            "data/processed/payments.csv",
            "data/processed/products.csv",
            "data/processed/reviews.csv",
            "data/processed/sellers.csv",
            "data/processed/translation.csv",
        ],
    },
    {
        "script":  SPRINT2_DIR / "02_feature_engineering.py",
        "label":   "02_feature_engineering",
        "outputs": ["data/master/master_table.csv"],
    },
    {
        "script":  SPRINT2_DIR / "03_limpieza.py",
        "label":   "03_limpieza",
        "outputs": ["data/master/master_table_limpia.csv"],
    },
    {
        "script":  SPRINT2_DIR / "04_split.py",
        "label":   "04_split",
        "outputs": [
            "data/master/X_train.csv",
            "data/master/y_train.csv",
            "data/master/X_val.csv",
            "data/master/y_val.csv",
            "data/master/X_backtest.csv",
            "data/master/y_backtest.csv",
            "data/master/X_live.csv",
            "data/master/y_live.csv",
        ],
    },
    # ── Sprint 3: modelado ───────────────────────────────────────────────────
    {
        "script":  SPRINT_DIR / "05_baseline.py",
        "label":   "05_baseline",
        "outputs": [
            "data/reportes/resultados_baseline.json",
            "data/reportes/estado_baseline.json",
        ],
    },
    {
        "script":  SPRINT_DIR / "06_optuna.py",
        "label":   "06_optuna",
        "outputs": [
            "data/reportes/resultados_optuna.json",
            "data/reportes/mejores_params.json",
            "data/reportes/modelos_optuna.pkl",
            "data/reportes/estudios_optuna.pkl",
            "data/reportes/estado_optuna.json",
        ],
    },
    {
        "script":  SPRINT_DIR / "07_graficos.py",
        "label":   "07_graficos",
        "outputs": [
            "reports/01_comparacion_metricas.png",
            "reports/02_roc_val.png",
            "reports/03_roc_backtest.png",
            "reports/04_pr_val.png",
            "reports/05_pr_backtest.png",
            "reports/06_confusion_val.png",
            "reports/07_confusion_backtest.png",
            "reports/08_dist_probabilidades.png",
            "reports/09_optuna_convergencia.png",
            "reports/10_optuna_hiperparametros.png",
            "reports/11_feature_importance_nativa.png",
            "reports/12_feature_importance_comparativo.png",
        ],
    },
]


def verificar_archivos(rutas: list[str]) -> bool:
    todos_ok = True
    for ruta in rutas:
        if not (ROOT / ruta).exists():
            print(f"  ⚠️  No encontrado: {ruta}")
            todos_ok = False
    return todos_ok


def main():
    print("🔍 Verificando archivos raw...")
    if not verificar_archivos(RAW_ESPERADOS):
        print("\n❌ Faltan archivos fuente en data/raw/. Revisá la carpeta antes de continuar.")
        sys.exit(1)
    print("✅ Archivos raw OK\n")

    print("🚀 Iniciando pipeline Sprint 3\n")
    tiempo_total = time.time()

    for etapa in PIPELINE:
        script  = etapa["script"]
        label   = etapa["label"]
        outputs = etapa["outputs"]

        print(f"{'='*55}")
        print(f"▶  {label}")
        print(f"{'='*55}")

        t0 = time.time()
        resultado = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=False,
        )
        elapsed = time.time() - t0

        if resultado.returncode != 0:
            print(f"\n❌ {label} falló (returncode={resultado.returncode}).")
            print("   Pipeline detenido.")
            sys.exit(1)

        if not verificar_archivos(outputs):
            print(f"\n❌ {label} terminó pero no generó todos los outputs esperados.")
            print("   Pipeline detenido.")
            sys.exit(1)

        print(f"✅ {label} OK  ⏱  {elapsed:.1f}s\n")

    total = time.time() - tiempo_total
    print(f"🎉 Pipeline Sprint 3 completo en {total:.1f}s\n")


if __name__ == "__main__":
    main()