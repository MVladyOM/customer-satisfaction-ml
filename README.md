# Pipeline de Predicción de Satisfacción del Cliente — Olist
**Grupo 1 · Sprint 2**

---

## Descripción General

Este proyecto construye un pipeline reproducible y escalable para predecir si un cliente de Olist quedará satisfecho con su pedido, definido como `review_score >= 4`. El pipeline se ejecuta mensualmente incorporando nuevos datos y genera todos los artefactos necesarios para el entrenamiento y evaluación de modelos.

**Variable target:** `satisfecho` (binaria: 1 = satisfecho, 0 = insatisfecho)  
**Umbral de satisfacción:** `review_score >= 4`  
**Dataset fuente:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 9 tablas CSV

---

## Estructura del Proyecto

```
proyecto/
├── data/
│   ├── raw/                        # Datos originales sin modificar (9 CSVs)
│   ├── processed/                  # Tablas cargadas y validadas (Script 01)
│   ├── master/                     # Master table y splits de entrenamiento
│   └── reportes/                   # JSONs y CSVs de métricas y estado
│
├── 00_run_pipeline.py              # Orquestador principal (ejecuta todo)
├── 01_carga_datos.py               # Carga y validación de calidad
├── 02_feature_engineering.py       # Construcción de 68 variables
├── 03_limpieza.py                  # Limpieza y preparación de la master table
├── 04_split.py                     # Split temporal + transformaciones + selección
└── README.md
```

---

## Requisitos

```bash
pip install pandas numpy scikit-learn
```

**Python:** 3.10 o superior (se usa `list[str]` y `tuple[...]` como type hints)

### Archivos raw necesarios en `data/raw/`

| Archivo | Descripción |
|---|---|
| `olist_orders_dataset.csv` | Pedidos y fechas |
| `olist_order_reviews_dataset.csv` | Reviews y scores |
| `olist_order_items_dataset.csv` | Ítems por pedido |
| `olist_order_payments_dataset.csv` | Pagos |
| `olist_customers_dataset.csv` | Clientes |
| `olist_products_dataset.csv` | Productos |
| `olist_sellers_dataset.csv` | Vendedores |
| `olist_geolocation_dataset.csv` | Geolocalización |
| `product_category_name_translation.csv` | Traducción de categorías |

---

## Ejecución

### Opción 1 — Pipeline completo (recomendado)

```bash
python 00_run_pipeline.py
```

El orquestador verifica los archivos raw, ejecuta los 4 scripts en orden y valida que cada uno genere sus outputs antes de continuar. Si algún script falla, el pipeline se detiene con un mensaje de error claro.

### Opción 2 — Script individual

```bash
python 01_carga_datos.py
python 02_feature_engineering.py
python 03_limpieza.py
python 04_split.py
```

Cada script depende del output del anterior. Deben ejecutarse en orden.

---

## Flujo de Datos

```
data/raw/ (9 CSVs)
    │
    ▼
Script 01 — Carga de Datos
    ├── Verifica shape, nulos, duplicados
    ├── Parsea columnas de fecha
    └── → data/processed/ (9 CSVs) + estado_carga.json
    │
    ▼
Script 02 — Feature Engineering
    ├── Filtra pedidos: status=delivered + tiene review
    ├── Construye 68 variables en 8 dominios (A–H)
    ├── Crea variable target: satisfecho
    ├── Aplica winsorización (percentiles 1%–99%)
    └── → data/master/master_table.csv + estado_feature_engineering.json
    │
    ▼
Script 03 — Limpieza
    ├── Elimina IDs, fechas crudas, leakage y duplicados
    ├── Elimina columnas con > 15% de nulos
    ├── Imputa nulos restantes (mediana / 'desconocido')
    ├── Agrupa categorías raras (< 1% frecuencia) → 'otros'
    └── → data/master/master_table_limpia.csv + estado_limpieza.json
    │
    ▼
Script 04 — Split + Transformaciones + Selección
    ├── Split temporal estricto (sin shuffle):
    │       Train    : sep 2016 – dic 2017
    │       Val      : ene 2018 – mar 2018
    │       Backtest : abr 2018 – jun 2018
    │       Live     : jul 2018 – oct 2018
    ├── Target Encoding (fit solo en train)
    ├── StandardScaler (fit solo en train)
    └── Selección de variables en 6 pasos (Pasos 0–6)
        └── → data/master/ (8 CSVs: X/y × 4 splits) + reportes
```

---

## Variables Generadas (68 features)

Las features se organizan en 8 dominios:

### A — Tiempo / Logística
| Variable | Descripción |
|---|---|
| `actual_delivery_days` | Días reales de entrega (compra → entrega) |
| `delivery_delay_days` | Días de retraso (negativo = entregado antes) |
| `delivered_on_time` | Flag: entregado antes o en la fecha prometida |
| `promised_delivery_days` | Días prometidos de entrega |
| `approval_time_hours` | Tiempo de aprobación en horas |
| `dispatch_time_hours` | Tiempo de despacho al carrier en horas |
| `delay_ratio` | Retraso / días prometidos |
| `purchase_month` | Mes de compra |
| `purchase_dayofweek` | Día de la semana de compra |
| `purchase_hour` | Hora de compra |

### B — Valor Económico
| Variable | Descripción |
|---|---|
| `total_price` | Precio total de los ítems |
| `total_freight_value` | Costo de flete total |
| `total_order_value` | Precio + flete |
| `avg_price_per_item` | Precio promedio por ítem |
| `max_price_item` | Precio máximo de un ítem |
| `min_price_item` | Precio mínimo de un ítem |
| `payment_value` | Valor total del pago |
| `freight_ratio` | Flete / (precio + flete) |

### C — Complejidad del Pedido
| Variable | Descripción |
|---|---|
| `order_item_count` | Cantidad de ítems |
| `unique_sellers` | Número de vendedores distintos |
| `is_multi_seller` | Flag: más de un vendedor |
| `is_multi_item` | Flag: más de un ítem |
| `payment_installments` | Cuotas del pago |
| `pago_en_cuotas` | Flag: pago en cuotas |
| `log_payment_value` | Log del valor del pago |
| `log_freight_value` | Log del flete |

### D — Vendedor
| Variable | Descripción |
|---|---|
| `seller_state` | Estado del vendedor |
| `seller_customer_same_state` | Flag: vendedor y cliente en el mismo estado |

### E — Producto / Categoría
| Variable | Descripción |
|---|---|
| `product_category_name_english` | Categoría del producto (en inglés) |
| `product_weight_g` | Peso del producto (gramos) |
| `product_volume_cm3` | Volumen del producto (cm³) |
| `product_photos_qty` | Cantidad de fotos en el listado |
| `has_product_photo` | Flag: tiene al menos una foto |
| `product_description_lenght` | Largo de la descripción |
| `product_name_lenght` | Largo del nombre |

### F — Cliente / Geografía
| Variable | Descripción |
|---|---|
| `customer_state` | Estado del cliente |
| `customer_region` | Región de Brasil (nordeste / norte / sudeste / sul / centro_oeste) |
| `customer_high_risk_state` | Flag: estado con baja satisfacción histórica (RR, AL, MA, SE, PA) |

### G — Pago
| Variable | Descripción |
|---|---|
| `payment_type` | Tipo de pago dominante |
| `payment_types_count` | Número de tipos de pago distintos |

### H — Variables de Interacción
| Variable | Descripción |
|---|---|
| `interaccion_retraso_items` | `delivery_delay_days × order_item_count` |
| `interaccion_precio_tarde` | `payment_value × es_tarde` |
| `interaccion_entrega_flete` | `actual_delivery_days × freight_ratio` |

---

## Limpieza Aplicada (Script 03)

### Columnas eliminadas

| Categoría | Columnas |
|---|---|
| IDs (sin poder predictivo) | `order_id`, `customer_id`, `seller_id`, `product_id` |
| Fechas crudas | `order_approved_at`, `order_delivered_*`, `order_estimated_*`, `purchase_year_month` |
| Leakage directo del target | `review_score`, `review_comment_*`, `has_comment`, `has_review_title`, `comment_length` |
| Duplicadas / redundantes | `uses_installments`, `customer_city`, `seller_city` |

> **Nota sobre leakage:** `review_score` es el target sin binarizar. Los campos de comentario (`review_comment_message`, `review_comment_title`) son escritos por el cliente *después* de haber decidido su score, por lo que incluirlos equivale a mirar el futuro. Se eliminan rigurosamente.

### Imputación de nulos
- **Numéricas:** mediana (robusta ante distribuciones sesgadas)
- **Categóricas:** `'desconocido'` (preserva la ausencia como señal)

### Agrupación de categorías raras
Categorías con frecuencia relativa < 1% se reemplazan por `'otros'` en: `product_category_name_english`, `customer_state`, `seller_state`, `payment_type`, `customer_region`.

---

## Selección de Variables (Script 04)

La selección se realiza en 6 pasos secuenciales. Cada paso prueba sus umbrales en paralelo y elige el que maximiza `AUC_val`, siempre que no degrade el AUC más de **0.005 pp** respecto al estado anterior. Si ningún umbral mejora, el paso se omite.

| Paso | Método | Umbrales / Variantes |
|---|---|---|
| 0 | Estado inicial (baseline) | — |
| 1 | Missing variable | 0.10, 0.15 |
| 2 | PSI (Population Stability Index) | > 0.20, bins=10 |
| 3 | Correlación | 0.80, 0.90, 0.95, 0.99 |
| 4 | Univariante (AUC individual) | 0.01, 0.02, 0.05 |
| 5 | Variance Threshold | 0.01 |
| 6 | RF Importance (top-N) | top-20, top-25, top-30 |

**Regla de oro:** Backtest y Live **nunca se tocan** durante la selección de variables. Solo reciben los transformadores ya fiteados en train.

---

## Split Temporal

| Conjunto | Período | Uso |
|---|---|---|
| **Train** | sep 2016 – dic 2017 | Fit de modelos y transformadores |
| **Val** | ene 2018 – mar 2018 | Selección de variables y ajuste de hiperparámetros |
| **Backtest** | abr 2018 – jun 2018 | Evaluación fuera de muestra |
| **Live** | jul 2018 – oct 2018 | Simulación de producción |

El split es **temporal y sin shuffle** para respetar la naturaleza secuencial de los datos y evitar data leakage temporal.

---

## Outputs Generados

### Datos
| Archivo | Descripción |
|---|---|
| `data/processed/<tabla>.csv` | 9 tablas validadas y parseadas |
| `data/master/master_table.csv` | Tabla con todas las features (pre-limpieza) |
| `data/master/master_table_limpia.csv` | Tabla lista para modelar |
| `data/master/X_train.csv` / `y_train.csv` | Conjunto de entrenamiento |
| `data/master/X_val.csv` / `y_val.csv` | Conjunto de validación |
| `data/master/X_backtest.csv` / `y_backtest.csv` | Conjunto de backtest |
| `data/master/X_live.csv` / `y_live.csv` | Conjunto live |

### Reportes
| Archivo | Descripción |
|---|---|
| `data/reportes/estado_carga.json` | Shape, nulos y duplicados por tabla |
| `data/reportes/estado_feature_engineering.json` | Shape final, distribución del target, nulos |
| `data/reportes/estado_limpieza.json` | Log de cada paso de limpieza |
| `data/reportes/features_seleccionadas.json` | Lista final de features seleccionadas |
| `data/reportes/tabla_seleccion_variables.csv` | Métricas por paso/umbral (AUC, Gini, F1, etc.) |
| `data/reportes/tabla_detalle_eliminadas.csv` | Detalle de cada feature eliminada y en qué paso |
| `data/reportes/estado_seleccion.json` | Reporte completo del Script 04 |

---

## Métricas de Evaluación

El pipeline reporta las siguientes métricas en cada paso de selección de variables:

| Métrica | Descripción |
|---|---|
| **AUC ROC** | Área bajo la curva ROC — métrica principal |
| **Gini** | `2 × AUC - 1` — mide poder discriminante |
| **Accuracy** | Exactitud global |
| **Recall** | Sensibilidad (detección de insatisfechos) |
| **Precision** | Precisión en las predicciones positivas |
| **F1 Score** | Media armónica de Precision y Recall |

Todas se calculan tanto en **train** como en **val** para monitorear overfitting.

---

## Decisiones de Diseño

**¿Por qué split temporal y no aleatorio?**  
Los pedidos tienen fecha. Un split aleatorio permitiría que el modelo "vea" compras de noviembre 2017 al entrenar y luego prediga sobre compras de octubre 2017, lo que es imposible en producción. El split temporal replica fielmente el escenario real.

**¿Por qué Target Encoding y no One-Hot?**  
`product_category_name_english` tiene alta cardinalidad. One-Hot generaría decenas de columnas dispersas. Target Encoding comprime la información en una sola variable numérica y es especialmente efectivo para variables con fuerte relación con el target. El fit se realiza **solo en train** para evitar leakage.

**¿Por qué winsorizar?**  
Variables como `delivery_delay_days` o `payment_value` tienen distribuciones muy sesgadas con outliers extremos. La winsorización al percentil 1%–99% reduce su influencia sin eliminar los registros.

**¿Por qué mediana para imputar y no media?**  
La mayoría de las variables numéricas del dataset Olist tienen distribuciones asimétricas. La mediana es robusta ante outliers y preserva mejor la tendencia central real.

---

## Incorporación Mensual de Nuevos Datos

Para simular la ejecución mensual con nuevos datos:

1. Agregar los nuevos CSVs en `data/raw/` (reemplazando o extendiendo los existentes)
2. Ejecutar `python 00_run_pipeline.py`
3. Los transformadores (Target Encoding, StandardScaler) se refitean sobre el nuevo train
4. Los reportes en `data/reportes/` se sobreescriben con el estado actualizado

> En un entorno productivo real, se recomienda versionar los outputs con timestamps para mantener historial de ejecuciones.

---

## Autores

Grupo 1 — Sprint 2  
Proyecto: Predicción de Satisfacción del Cliente — Olist
