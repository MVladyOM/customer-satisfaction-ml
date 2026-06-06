"""
=============================================================================
Script 02 - Feature Engineering & Master Table
Proyecto : Predicción de Satisfacción del Cliente — Olist
Grupo 1  | Sprint 2
=============================================================================
Responsabilidad:
  - Leer tablas desde data/processed
  - Construir la Master Table: 1 fila = 1 pedido (order_id)
  - Generar 52+ features organizadas en dominios:
      A. Tiempo / Logística
      B. Valor económico
      C. Complejidad del pedido
      D. Vendedor
      E. Producto / Categoría
      F. Cliente / Geografía
      G. Pago
      H. Variables de texto (review)
  - Aplicar winsorización (clipado por percentiles)
  - Las variables categóricas se dejan en crudo — el Target Encoding
    se realiza en Script 04 SOLO sobre el conjunto train (sin leakage)
  - Guardar master_table.csv en data/master

Salida:
  data/master/master_table.csv
  data/reportes/estado_feature_engineering.json
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
PROCESSED_PATH = os.path.join("data", "processed")
MASTER_PATH    = os.path.join("data", "master")
REPORTES_PATH  = os.path.join("data", "reportes")

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
REVIEW_SCORE_UMBRAL = 4          # satisfecho si review_score >= 4
WINSOR_LOWER        = 0.01       # percentil inferior para winsorización
WINSOR_UPPER        = 0.99       # percentil superior para winsorización


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------

def cargar_processed() -> dict:
    """Lee todas las tablas desde data/processed."""
    nombres = [
        "orders", "reviews", "items", "payments",
        "customers", "products", "sellers", "translation",
    ]
    datos = {}
    print("\n[1] Cargando tablas procesadas …")
    for nombre in nombres:
        ruta = os.path.join(PROCESSED_PATH, f"{nombre}.csv")
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No se encontró {ruta}. "
                "Ejecuta primero el Script 01."
            )
        datos[nombre] = pd.read_csv(ruta, low_memory=False)
        print(f"  {nombre:15s}: {datos[nombre].shape}")
    return datos


# ---------------------------------------------------------------------------
# 2. PARSEO DE FECHAS
# ---------------------------------------------------------------------------

def parsear_fechas(datos: dict) -> dict:
    """Convierte columnas de fecha a datetime."""
    cols_orders = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in cols_orders:
        if col in datos["orders"].columns:
            datos["orders"][col] = pd.to_datetime(
                datos["orders"][col], errors="coerce"
            )

    for col in ["review_creation_date", "review_answer_timestamp"]:
        if col in datos["reviews"].columns:
            datos["reviews"][col] = pd.to_datetime(
                datos["reviews"][col], errors="coerce"
            )

    if "shipping_limit_date" in datos["items"].columns:
        datos["items"]["shipping_limit_date"] = pd.to_datetime(
            datos["items"]["shipping_limit_date"], errors="coerce"
        )
    return datos


# ---------------------------------------------------------------------------
# 3. BASE: POBLAR OBJETO CON PEDIDOS ENTREGADOS CON REVIEW
# ---------------------------------------------------------------------------

def construir_base(datos: dict) -> pd.DataFrame:
    """
    Población objetivo:
      - Pedidos con order_status == 'delivered'
      - Con review_score registrado
    1 fila = 1 pedido (order_id)
    """
    print("\n[2] Construyendo base de pedidos …")

    orders = datos["orders"].copy()

    # Solo pedidos entregados
    orders = orders[orders["order_status"] == "delivered"].copy()
    print(f"  Pedidos delivered          : {len(orders):,}")

    # Agregar review (1 review por pedido — tomar la más reciente si hay duplicados)
    reviews = (
        datos["reviews"]
        .sort_values("review_answer_timestamp", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")
        [["order_id", "review_score", "review_comment_message", "review_comment_title"]]
    )

    base = orders.merge(reviews, on="order_id", how="inner")
    print(f"  Pedidos con review_score   : {len(base):,}")
    return base


# ---------------------------------------------------------------------------
# 4. DOMINIO A — TIEMPO / LOGÍSTICA
# ---------------------------------------------------------------------------

def features_tiempo(base: pd.DataFrame) -> pd.DataFrame:
    print("  → Dominio A: Tiempo / Logística")

    b = base.copy()

    # Días reales de entrega (purchase → delivered)
    b["actual_delivery_days"] = (
        b["order_delivered_customer_date"] - b["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Días de retraso (negativo = entregado antes)
    b["delivery_delay_days"] = (
        b["order_delivered_customer_date"] - b["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    # Entregado a tiempo (flag binario)
    b["delivered_on_time"] = (b["delivery_delay_days"] <= 0).astype(int)

    # Días prometidos (purchase → estimated)
    b["promised_delivery_days"] = (
        b["order_estimated_delivery_date"] - b["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # Tiempo de aprobación (purchase → approved)
    b["approval_time_hours"] = (
        b["order_approved_at"] - b["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    # Tiempo de despacho (approved → carrier)
    b["dispatch_time_hours"] = (
        b["order_delivered_carrier_date"] - b["order_approved_at"]
    ).dt.total_seconds() / 3600

    # Ratio retraso / días prometidos
    b["delay_ratio"] = b["delivery_delay_days"] / b["promised_delivery_days"].replace(0, np.nan)

    # Variables temporales del mes de compra
    b["purchase_month"] = b["order_purchase_timestamp"].dt.month
    b["purchase_dayofweek"] = b["order_purchase_timestamp"].dt.dayofweek
    b["purchase_hour"] = b["order_purchase_timestamp"].dt.hour
    b["purchase_year_month"] = b["order_purchase_timestamp"].dt.to_period("M").astype(str)

    return b


# ---------------------------------------------------------------------------
# 5. DOMINIO B — VALOR ECONÓMICO
# ---------------------------------------------------------------------------

def features_valor(base: pd.DataFrame, items: pd.DataFrame,
                   payments: pd.DataFrame) -> pd.DataFrame:
    print("  → Dominio B: Valor Económico")

    # --- Items: agrupar por order_id ---
    items_agg = items.groupby("order_id").agg(
        order_item_count    = ("order_item_id", "count"),
        total_price         = ("price", "sum"),
        total_freight_value = ("freight_value", "sum"),
        avg_price_per_item  = ("price", "mean"),
        max_price_item      = ("price", "max"),
        min_price_item      = ("price", "min"),
        unique_sellers      = ("seller_id", "nunique"),
    ).reset_index()

    items_agg["freight_ratio"] = (
        items_agg["total_freight_value"]
        / (items_agg["total_price"] + items_agg["total_freight_value"]).replace(0, np.nan)
    )

    # --- Payments: agrupar por order_id ---
    payments_agg = payments.groupby("order_id").agg(
        payment_value        = ("payment_value", "sum"),
        payment_installments = ("payment_installments", "max"),
        payment_types_count  = ("payment_type", "nunique"),
    ).reset_index()

    # Tipo de pago dominante (el de mayor valor)
    pay_type = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")[["order_id", "payment_type"]]
    )
    payments_agg = payments_agg.merge(pay_type, on="order_id", how="left")

    base = base.merge(items_agg,    on="order_id", how="left")
    base = base.merge(payments_agg, on="order_id", how="left")

    # Valor total del pedido (precio + flete)
    base["total_order_value"] = base["total_price"] + base["total_freight_value"]

    return base


# ---------------------------------------------------------------------------
# 6. DOMINIO C — COMPLEJIDAD DEL PEDIDO
# ---------------------------------------------------------------------------

def features_complejidad(base: pd.DataFrame) -> pd.DataFrame:
    print("  → Dominio C: Complejidad del Pedido")

    # Pedido multi-seller
    base["is_multi_seller"] = (base["unique_sellers"] > 1).astype(int)

    # Pedido multi-item
    base["is_multi_item"] = (base["order_item_count"] > 1).astype(int)

    # Pago en cuotas
    base["uses_installments"] = (base["payment_installments"] > 1).astype(int)

    # Pago en cuotas (flag)
    base["pago_en_cuotas"] = (base["payment_installments"].fillna(1) > 1).astype(int)

    # Precio promedio por item
    base["precio_por_item"] = (
        base["total_price"] / base["order_item_count"].replace(0, np.nan)
    )

    # Log del valor (reduce asimetría)
    base["log_payment_value"] = np.log1p(base["payment_value"].fillna(0))
    base["log_freight_value"] = np.log1p(base["total_freight_value"].fillna(0))

    return base


# ---------------------------------------------------------------------------
# 7. DOMINIO D — VENDEDOR
# ---------------------------------------------------------------------------

def features_vendedor(base: pd.DataFrame, items: pd.DataFrame,
                       sellers: pd.DataFrame) -> pd.DataFrame:
    print("  → Dominio D: Vendedor")

    # Vendedor principal del pedido (mayor precio)
    vendedor_principal = (
        items.sort_values("price", ascending=False)
        .drop_duplicates("order_id")[["order_id", "seller_id"]]
    )
    base = base.merge(vendedor_principal, on="order_id", how="left")
    base = base.merge(
        sellers[["seller_id", "seller_state", "seller_city"]],
        on="seller_id", how="left"
    )

    # Nota: seller_customer_same_state se calcula en features_cliente()
    # porque customer_state se agrega en ese paso posterior.
    return base


# ---------------------------------------------------------------------------
# 8. DOMINIO E — PRODUCTO / CATEGORÍA
# ---------------------------------------------------------------------------

def features_producto(base: pd.DataFrame, items: pd.DataFrame,
                       products: pd.DataFrame,
                       translation: pd.DataFrame) -> pd.DataFrame:
    print("  → Dominio E: Producto / Categoría")

    # Traducir categorías
    products = products.merge(translation, on="product_category_name", how="left")

    # Producto principal (mayor precio)
    prod_principal = (
        items.sort_values("price", ascending=False)
        .drop_duplicates("order_id")[["order_id", "product_id"]]
    )
    base = base.merge(prod_principal, on="order_id", how="left")
    base = base.merge(
        products[[
            "product_id",
            "product_category_name_english",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
            "product_photos_qty",
            "product_description_lenght",
            "product_name_lenght",
        ]],
        on="product_id", how="left"
    )

    # Volumen del producto (cm³)
    base["product_volume_cm3"] = (
        base["product_length_cm"]
        * base["product_height_cm"]
        * base["product_width_cm"]
    )

    # Tiene fotos
    base["has_product_photo"] = (base["product_photos_qty"].fillna(0) > 0).astype(int)

    return base


# ---------------------------------------------------------------------------
# 9. DOMINIO F — CLIENTE / GEOGRAFÍA
# ---------------------------------------------------------------------------

def features_cliente(base: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    print("  → Dominio F: Cliente / Geografía")

    base = base.merge(
        customers[["customer_id", "customer_state", "customer_city"]],
        on="customer_id", how="left"
    )

    # Regiones de Brasil
    nordeste = {"MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"}
    norte    = {"AM", "RR", "AP", "PA", "TO", "RO", "AC"}
    sudeste  = {"SP", "RJ", "MG", "ES"}
    sul      = {"PR", "SC", "RS"}
    co       = {"GO", "MT", "MS", "DF"}

    def mapear_region(estado):
        if estado in nordeste: return "nordeste"
        if estado in norte:    return "norte"
        if estado in sudeste:  return "sudeste"
        if estado in sul:      return "sul"
        if estado in co:       return "centro_oeste"
        return "outro"

    base["customer_region"] = base["customer_state"].apply(mapear_region)

    # Flag estados de baja satisfacción (confirmados en EDA Sprint 1)
    estados_riesgo = {"RR", "AL", "MA", "SE", "PA"}
    base["customer_high_risk_state"] = base["customer_state"].isin(estados_riesgo).astype(int)

    # Vendedor y cliente en el mismo estado (ahora customer_state ya existe)
    if "seller_state" in base.columns:
        base["seller_customer_same_state"] = (
            base["seller_state"] == base["customer_state"]
        ).astype(int)

    return base


# ---------------------------------------------------------------------------
# 10. DOMINIO G — TEXTO / REVIEW
# ---------------------------------------------------------------------------

def features_texto(base: pd.DataFrame) -> pd.DataFrame:
    print("  → Dominio G: Variables de Texto (Review)")

    # Tiene comentario escrito
    base["has_comment"] = (
        base["review_comment_message"].notna()
        & (base["review_comment_message"].astype(str).str.strip() != "")
    ).astype(int)

    # Tiene título en el review
    base["has_review_title"] = (
        base["review_comment_title"].notna()
        & (base["review_comment_title"].astype(str).str.strip() != "")
    ).astype(int)

    # Largo del comentario (si existe)
    base["comment_length"] = (
        base["review_comment_message"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.len()
    )

    return base


# ---------------------------------------------------------------------------
# 11. DOMINIO H — VARIABLES DE INTERACCIÓN
# ---------------------------------------------------------------------------

def features_interaccion(base: pd.DataFrame) -> pd.DataFrame:
    """
    Capturan efectos conjuntos que ninguna variable sola puede explicar.
    Un modelo lineal no detecta estas relaciones — hay que crearlas
    explícitamente.

    - interaccion_retraso_items : dias_retraso × total_items
      Pedido con muchos productos Y retraso → doble insatisfacción.
    - interaccion_precio_tarde  : payment_value × es_tarde
      Cliente que pagó más y recibió tarde → caso más crítico.
      Captura directamente la paradoja del precio confirmada en EDA.
    - interaccion_entrega_flete : actual_delivery_days × freight_ratio
      Entrega larga con flete proporcionalmente caro → percepción de
      haber pagado mucho por un mal servicio.
    """
    print("  → Dominio H: Variables de Interacción")

    # es_tarde = 1 si delivery_delay_days > 0
    es_tarde = (base["delivery_delay_days"].fillna(0) > 0).astype(int)

    base["interaccion_retraso_items"] = (
        base["delivery_delay_days"].fillna(0) * base["order_item_count"].fillna(1)
    )
    base["interaccion_precio_tarde"] = (
        base["payment_value"].fillna(0) * es_tarde
    )
    base["interaccion_entrega_flete"] = (
        base["actual_delivery_days"].fillna(0) * base["freight_ratio"].fillna(0)
    )

    return base


# ---------------------------------------------------------------------------
# 12. TARGET
# ---------------------------------------------------------------------------

def crear_target(base: pd.DataFrame) -> pd.DataFrame:
    print("\n[4] Creando variable target …")

    base["review_score"] = pd.to_numeric(base["review_score"], errors="coerce")
    base = base.dropna(subset=["review_score"]).copy()
    base["satisfecho"] = (base["review_score"] >= REVIEW_SCORE_UMBRAL).astype(int)

    dist = base["satisfecho"].value_counts(normalize=True)
    print(f"  Satisfecho (1)   : {dist.get(1, 0):.1%}")
    print(f"  Insatisfecho (0) : {dist.get(0, 0):.1%}")
    print(f"  Total registros  : {len(base):,}")
    return base


# ---------------------------------------------------------------------------
# 12. WINSORIZACIÓN
# ---------------------------------------------------------------------------

COLS_WINSORIZAR = [
    "actual_delivery_days", "delivery_delay_days", "approval_time_hours",
    "dispatch_time_hours", "total_price", "total_freight_value",
    "payment_value", "freight_ratio", "delay_ratio",
    "product_weight_g", "product_volume_cm3", "comment_length",
    "precio_por_item", "interaccion_retraso_items",
    "interaccion_precio_tarde", "interaccion_entrega_flete",
]

def winsorizacion(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[5] Aplicando winsorización …")
    for col in COLS_WINSORIZAR:
        if col not in df.columns:
            continue
        lo = df[col].quantile(WINSOR_LOWER)
        hi = df[col].quantile(WINSOR_UPPER)
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ---------------------------------------------------------------------------
# 13. GUARDAR
# ---------------------------------------------------------------------------

def guardar_master(df: pd.DataFrame) -> None:
    os.makedirs(MASTER_PATH,   exist_ok=True)
    os.makedirs(REPORTES_PATH, exist_ok=True)

    ruta = os.path.join(MASTER_PATH, "master_table.csv")
    df.to_csv(ruta, index=False)
    print(f"\n  Master Table guardada → {ruta}")
    print(f"  Shape final: {df.shape[0]:,} filas × {df.shape[1]} columnas")


def guardar_estado(df: pd.DataFrame, inicio: datetime) -> None:
    reporte = {
        "script"        : "02_feature_engineering",
        "timestamp"     : datetime.now().isoformat(),
        "duracion_seg"  : round((datetime.now() - inicio).total_seconds(), 2),
        "shape"         : {"filas": int(df.shape[0]), "columnas": int(df.shape[1])},
        "target"        : {
            "satisfecho"   : int(df["satisfecho"].sum()),
            "insatisfecho" : int((df["satisfecho"] == 0).sum()),
            "pct_satisfecho": round(df["satisfecho"].mean(), 4),
        },
        "nulos_por_columna": {
            col: int(n) for col, n in df.isnull().sum().items() if n > 0
        },
        "columnas": list(df.columns),
    }
    ruta = os.path.join(REPORTES_PATH, "estado_feature_engineering.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Reporte guardado  → {ruta}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    inicio = datetime.now()
    print("=" * 65)
    print("SCRIPT 02 — FEATURE ENGINEERING & MASTER TABLE")
    print(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # 1. Carga
    datos = cargar_processed()
    datos = parsear_fechas(datos)

    # 2. Base
    base = construir_base(datos)

    # 3. Features por dominio
    print("\n[3] Generando features por dominio …")
    base = features_tiempo(base)
    base = features_valor(base, datos["items"], datos["payments"])
    base = features_complejidad(base)
    base = features_vendedor(base, datos["items"], datos["sellers"])
    base = features_producto(base, datos["items"], datos["products"], datos["translation"])
    base = features_cliente(base, datos["customers"])
    base = features_texto(base)
    base = features_interaccion(base)

    # 4. Target
    base = crear_target(base)

    # 5. Winsorización
    base = winsorizacion(base)

    # 6. Guardar
    print("\n[6] Guardando resultados …")
    guardar_master(base)
    guardar_estado(base, inicio)

    print("\n" + "=" * 65)
    print(f"Script 02 completado en {(datetime.now()-inicio).total_seconds():.1f}s")
    print("=" * 65 + "\n")

    return base


if __name__ == "__main__":
    main()
