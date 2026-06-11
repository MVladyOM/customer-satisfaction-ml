import subprocess
import sys
import time
from pathlib import Path

# Directorio raíz del proyecto (un nivel arriba de sprint2/)
ROOT = Path(__file__).resolve().parent.parent
# Directorio donde viven los scripts del sprint
SPRINT_DIR = Path(__file__).resolve().parent

# Archivos raw que deben existir antes de arrancar
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

PIPELINE = [
    {
        "script": "01_carga_datos.py",
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
        "script": "02_feature_engineering.py",
        "outputs": ["data/master/master_table.csv"],
    },
    {
        "script": "03_limpieza.py",
        "outputs": ["data/master/master_table_limpia.csv"],
    },
    {
        "script": "04_split.py",
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
    {
        "script": "05_optuna.py",
        "outputs": [
            "data/reportes/resultados_optuna.json",
            "data/reportes/mejores_params.json",
            "data/reportes/modelos_optuna.pkl",
            "data/reportes/estudios_optuna.pkl",
            "data/reportes/estado_optuna.json",
        ],
    },
    {
        "script": "06_graficos.py",
        "outputs": [
            "reports/01_comparacion_metricas.png",
            "reports/09_optuna_convergencia.png",
            "reports/11_feature_importance_nativa.png",
            "reports/12_feature_importance_comparativo.png",
        ],
    },
]


def verificar_archivos(rutas: list[str], contexto: str) -> bool:
    todos_ok = True
    for ruta in rutas:
        if not (ROOT / ruta).exists():
            print(f"  ⚠️  No encontrado: {ruta}")
            todos_ok = False
    return todos_ok


def main():
    print("🔍 Verificando archivos raw...")
    if not verificar_archivos(RAW_ESPERADOS, "raw"):
        print("\n❌ Faltan archivos fuente en data/raw/. Revisá la carpeta antes de continuar.")
        sys.exit(1)
    print("✅ Archivos raw OK\n")

    print("🚀 Iniciando pipeline\n")
    tiempo_total = time.time()

    for etapa in PIPELINE:
        script  = etapa["script"]
        outputs = etapa["outputs"]

        print(f"{'='*55}")
        print(f"▶  {script}")
        print(f"{'='*55}")

        t0 = time.time()
        resultado = subprocess.run(
            [sys.executable, str(SPRINT_DIR / script)],
            cwd=ROOT,
            check=False,
        )
        elapsed = time.time() - t0

        if resultado.returncode != 0:
            print(f"\n❌ {script} falló (returncode={resultado.returncode}).")
            print("   Pipeline detenido.")
            sys.exit(1)

        if not verificar_archivos(outputs, script):
            print(f"\n❌ {script} terminó pero no generó todos los outputs esperados.")
            print("   Pipeline detenido.")
            sys.exit(1)

        print(f"✅ {script} OK  ⏱  {elapsed:.1f}s\n")

    total = time.time() - tiempo_total
    print(f"🎉 Pipeline completo en {total:.1f}s\n")


if __name__ == "__main__":
    main()