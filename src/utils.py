 # src/utils.py
import pandas as pd
from src.config import RAW_DIR, PROC_DIR

def load_raw_csvs() -> dict[str, pd.DataFrame]:
    """Carga los 9 CSVs de Olist como diccionario de DataFrames."""
    files = {
        "customers"   : "olist_customers_dataset.csv",
        "geolocation" : "olist_geolocation_dataset.csv",
        "order_items" : "olist_order_items_dataset.csv",
        "payments"    : "olist_order_payments_dataset.csv",
        "reviews"     : "olist_order_reviews_dataset.csv",
        "orders"      : "olist_orders_dataset.csv",
        "products"    : "olist_products_dataset.csv",
        "sellers"     : "olist_sellers_dataset.csv",
        "category"    : "product_category_name_translation.csv",
    }
    return {key: pd.read_csv(RAW_DIR / fname) for key, fname in files.items()}


def build_master_table(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Une las tablas principales en una Master Table centrada en order_id.
    Excluye geolocation (se trabaja por separado como feature geográfica).
    """

    # 1. Pagos agregados por orden (puede haber múltiples métodos de pago)
    payments_agg = (
        dfs["payments"]
        .groupby("order_id")
        .agg(
            payment_installments=("payment_installments", "max"),
            payment_value=("payment_value", "sum"),
            payment_type=("payment_type", "first"),
        )
        .reset_index()
    )

    # 2. Items agregados por orden
    items_agg = (
        dfs["order_items"]
        .groupby("order_id")
        .agg(
            order_item_count=("order_item_id", "count"),
            total_freight_value=("freight_value", "sum"),
            total_price=("price", "sum"),
            seller_id=("seller_id", "first"),
        )
        .reset_index()
    )

    # 3. Producto principal + categoría en inglés
    products = dfs["products"].merge(
        dfs["category"],
        on="product_category_name",
        how="left"
    )
    items_with_product = dfs["order_items"][["order_id", "product_id"]].drop_duplicates("order_id")
    items_with_product = items_with_product.merge(
        products[["product_id", "product_category_name_english",
                  "product_weight_g", "product_photos_qty"]],
        on="product_id",
        how="left"
    )

    # 4. Reviews
    reviews = dfs["reviews"][["order_id", "review_score","review_comment_message"]].drop_duplicates("order_id")

    # 5. Merge central
    master = (
        dfs["orders"]
        .merge(dfs["customers"][["customer_id", "customer_state", "customer_zip_code_prefix"]],
               on="customer_id", how="left")
        .merge(payments_agg,       on="order_id", how="left")
        .merge(items_agg,          on="order_id", how="left")
        .merge(items_with_product, on="order_id", how="left")
        .merge(reviews,            on="order_id", how="left")
    )

    return master


def save_master_table(master: pd.DataFrame) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(PROC_DIR / "master_table.csv", index=False)
    print(f"Master Table guardada: {master.shape[0]:,} filas x {master.shape[1]} columnas")
