"""
Stage 5 - Generador automático de informe de evaluación
=========================================================
Toma un CSV nuevo (con features + etiqueta real), aplica el modelo
entrenado (.pkl), calcula métricas y genera un informe HTML autocontenido
con fecha/hora de ejecución, gráficas interactivas (Plotly) y SHAP.

Uso:
    python 05_generar_reporte.py --csv nuevo_lote.csv --modelo modelo_rl.pkl --target satisfaccion

Requiere que el CSV ya esté preprocesado igual que en Stage 2
(mismas 13 features, mismo orden/encoding).
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from joblib import load
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shap
from jinja2 import Template
from sklearn.metrics import (
    accuracy_score, precision_recall_curve, precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix, average_precision_score
)

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_white"
HISTORICO_PATH = Path("historico_metricas.json")
SALIDA_DIR = Path("reportes")
SALIDA_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# CARGA Y PREDICCIÓN
# ──────────────────────────────────────────────────────────────────────────
def cargar_modelo(path_modelo: str):
    return load(path_modelo)


def predecir(modelo, df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y_true = df[target].values
    y_proba = modelo.predict_proba(X)[:, 1]
    y_pred = modelo.predict(X)
    return X, y_true, y_pred, y_proba


# ──────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ──────────────────────────────────────────────────────────────────────────
def calcular_metricas(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": acc,
        "auc_roc": auc,
        "clase_0": {"precision": prec[0], "recall": rec[0], "f1": f1[0], "support": int(support[0])},
        "clase_1": {"precision": prec[1], "recall": rec[1], "f1": f1[1], "support": int(support[1])},
    }


# ──────────────────────────────────────────────────────────────────────────
# GRÁFICAS (todas devuelven HTML embebible vía fig.to_html)
# ──────────────────────────────────────────────────────────────────────────
def fig_to_div(fig) -> str:
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      margin=dict(t=50, l=40, r=20, b=40))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def grafica_matriz_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm / cm.sum() * 100
    texto = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(2)]
             for i in range(2)]

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Pred: Insatisfecho", "Pred: Satisfecho"],
        y=["Real: Insatisfecho", "Real: Satisfecho"],
        text=texto, texttemplate="%{text}", colorscale="Blues", showscale=False
    ))
    fig.update_layout(title="Matriz de Confusión")
    return fig_to_div(fig)


def grafica_roc(y_true, y_proba, auc):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                  name=f"ROC (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(
        dash="dash", color="gray"), name="Azar"))
    fig.update_layout(title="Curva ROC", xaxis_title="Falsos Positivos",
                      yaxis_title="Verdaderos Positivos")
    return fig_to_div(fig)


def grafica_precision_recall(y_true, y_proba):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                  name=f"PR (AP={ap:.3f})"))
    fig.update_layout(title="Curva Precision-Recall",
                      xaxis_title="Recall", yaxis_title="Precision")
    return fig_to_div(fig)


def grafica_distribucion_proba(y_true, y_proba):
    fig = go.Figure()
    for clase, nombre, color in [(0, "Insatisfecho (real)", "indianred"), (1, "Satisfecho (real)", "seagreen")]:
        fig.add_trace(go.Histogram(
            x=y_proba[y_true == clase], name=nombre, opacity=0.6,
            marker_color=color, nbinsx=30
        ))
    fig.update_layout(title="Distribución de Probabilidades Predichas", barmode="overlay",
                      xaxis_title="P(satisfecho)", yaxis_title="Frecuencia")
    return fig_to_div(fig)


def _orden_por_riesgo(y_true, y_proba):
    """Ordena de mayor a menor riesgo de insatisfacción (proba baja = riesgo alto)."""
    orden = np.argsort(
        y_proba)  # ascendente: primero los de menor proba de satisfacción
    # 1 = insatisfecho, ordenado de más a menos riesgoso
    y_true_riesgo = (1 - y_true)[orden]
    return y_true_riesgo


def grafica_lift_chart(y_true, y_proba, n_deciles=10):
    y_riesgo = _orden_por_riesgo(y_true, y_proba)
    n = len(y_riesgo)
    tasa_base = y_riesgo.mean()

    deciles, lifts = [], []
    for i in range(1, n_deciles + 1):
        corte = int(n * i / n_deciles)
        tasa_decil = y_riesgo[:corte].mean()
        deciles.append(i * 10)
        lifts.append(tasa_decil / tasa_base if tasa_base > 0 else 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=deciles, y=lifts, name="Lift"))
    fig.add_hline(y=1, line_dash="dash", line_color="gray",
                  annotation_text="Selección al azar")
    fig.update_layout(title="Lift Chart — Priorización de clientes de riesgo",
                      xaxis_title="% de población contactada (ordenada por riesgo)",
                      yaxis_title="Lift (veces sobre el azar)")
    return fig_to_div(fig)


def grafica_cumulative_gains(y_true, y_proba):
    y_riesgo = _orden_por_riesgo(y_true, y_proba)
    n = len(y_riesgo)
    total_positivos = y_riesgo.sum()

    pct_poblacion = np.arange(1, n + 1) / n * 100
    pct_capturado = np.cumsum(y_riesgo) / total_positivos * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pct_poblacion, y=pct_capturado,
                  mode="lines", name="Modelo"))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(
        dash="dash", color="gray"), name="Azar"))
    fig.update_layout(title="Cumulative Gains Chart",
                      xaxis_title="% de población contactada", yaxis_title="% de insatisfechos capturados")
    return fig_to_div(fig)


def graficas_shap(modelo, X, max_muestras=500):
    """SHAP LinearExplainer: summary (beeswarm) + bar plot de importancia global."""
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    X_muestra = X.sample(min(max_muestras, len(X)), random_state=2357)
    maskerl = shap.maskers.Independent(X)
    explainer = shap.LinearExplainer(modelo, masker=maskerl)
    shap_values = explainer.shap_values(X_muestra)

    # Si el modelo devuelve lista (binaria con 2 clases), usar la clase positiva
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    def fig_a_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # Bar plot de importancia global
    fig1 = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_muestra, plot_type="bar",
                      show=False, plot_size=None)
    bar_b64 = fig_a_base64(fig1)

    # Beeswarm
    fig2 = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_muestra, show=False, plot_size=None)
    beeswarm_b64 = fig_a_base64(fig2)

    return beeswarm_b64, bar_b64


# ──────────────────────────────────────────────────────────────────────────
# HISTÓRICO DE MÉTRICAS
# ──────────────────────────────────────────────────────────────────────────
def actualizar_historico(metricas: dict, timestamp: str, n_registros: int):
    historico = []
    if HISTORICO_PATH.exists():
        historico = json.loads(HISTORICO_PATH.read_text())

    historico.append({
        "timestamp": timestamp,
        "n_registros": n_registros,
        "auc_roc": metricas["auc_roc"],
        "accuracy": metricas["accuracy"],
        "f1_clase_0": metricas["clase_0"]["f1"],
    })
    HISTORICO_PATH.write_text(json.dumps(historico, indent=2))
    return historico


def grafica_historico(historico: list):
    if len(historico) < 2:
        return None  # se maneja en la plantilla con un mensaje

    ts = [h["timestamp"] for h in historico]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=[h["auc_roc"] for h in historico], mode="lines+markers", name="AUC-ROC"))
    fig.add_trace(go.Scatter(x=ts, y=[
                  h["f1_clase_0"] for h in historico], mode="lines+markers", name="F1 Clase 0"))
    fig.update_layout(title="Tendencia histórica de métricas",
                      xaxis_title="Ejecución", yaxis_title="Valor")
    return fig_to_div(fig)


# ──────────────────────────────────────────────────────────────────────────
# PLANTILLA HTML
# ──────────────────────────────────────────────────────────────────────────
PLANTILLA = Template("""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe de Evaluación — {{ timestamp }}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1100px; margin: 40px auto; color: #222; }
  h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }
  h2 { margin-top: 50px; color: #2c3e50; }
  .meta { color: #666; font-size: 0.95em; margin-bottom: 30px; }
  .metricas { display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }
  .card { background: #f8f9fa; border-radius: 8px; padding: 16px 24px; min-width: 160px; border: 1px solid #e9ecef; }
  .card .valor { font-size: 1.8em; font-weight: bold; color: #2c3e50; }
  .card .etiqueta { font-size: 0.85em; color: #666; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; }
  th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: center; }
  th { background: #2c3e50; color: white; }
  .grafica { margin: 20px 0; }
  .aviso { background: #fff3cd; border: 1px solid #ffeeba; padding: 12px 18px; border-radius: 6px; color: #856404; }
  img { max-width: 100%; border-radius: 6px; }
</style>
</head>
<body>

<h1>📊 Informe de Evaluación del Modelo</h1>
<div class="meta">
  Generado el <strong>{{ timestamp }}</strong> · Archivo procesado: <strong>{{ nombre_csv }}</strong>
  · Registros: <strong>{{ n_registros }}</strong> · Modelo: <strong>{{ nombre_modelo }}</strong>
</div>

<h2>Resumen ejecutivo</h2>
<div class="metricas">
  <div class="card"><div class="valor">{{ "%.3f"|format(metricas.accuracy) }}</div><div class="etiqueta">Accuracy</div></div>
  <div class="card"><div class="valor">{{ "%.3f"|format(metricas.auc_roc) }}</div><div class="etiqueta">AUC-ROC</div></div>
  <div class="card"><div class="valor">{{ "%.3f"|format(metricas.clase_0.f1) }}</div><div class="etiqueta">F1 Clase 0 (Insatisfecho)</div></div>
  <div class="card"><div class="valor">{{ "%.3f"|format(metricas.clase_1.f1) }}</div><div class="etiqueta">F1 Clase 1 (Satisfecho)</div></div>
</div>

<table>
  <tr><th>Clase</th><th>Precision</th><th>Recall</th><th>F1</th><th>Soporte</th></tr>
  <tr><td>0 — Insatisfecho</td><td>{{ "%.3f"|format(metricas.clase_0.precision) }}</td><td>{{ "%.3f"|format(metricas.clase_0.recall) }}</td><td>{{ "%.3f"|format(metricas.clase_0.f1) }}</td><td>{{ metricas.clase_0.support }}</td></tr>
  <tr><td>1 — Satisfecho</td><td>{{ "%.3f"|format(metricas.clase_1.precision) }}</td><td>{{ "%.3f"|format(metricas.clase_1.recall) }}</td><td>{{ "%.3f"|format(metricas.clase_1.f1) }}</td><td>{{ metricas.clase_1.support }}</td></tr>
</table>

<h2>Evaluación del modelo</h2>
<div class="grafica">{{ g_matriz | safe }}</div>
<div class="grafica">{{ g_roc | safe }}</div>
<div class="grafica">{{ g_pr | safe }}</div>
<div class="grafica">{{ g_dist | safe }}</div>

<h2>Priorización para retención (Lift / Cumulative Gains)</h2>
<div class="grafica">{{ g_lift | safe }}</div>
<div class="grafica">{{ g_gains | safe }}</div>

<h2>Importancia de variables (SHAP)</h2>
<p>Calculado sobre una muestra de hasta 500 registros del lote actual.</p>
<img src="data:image/png;base64,{{ shap_bar }}" alt="SHAP bar plot">
<img src="data:image/png;base64,{{ shap_beeswarm }}" alt="SHAP beeswarm">

<h2>Tendencia histórica</h2>
{% if g_historico %}
  <div class="grafica">{{ g_historico | safe }}</div>
{% else %}
  <div class="aviso">Primera ejecución registrada en el histórico — la tendencia se mostrará a partir de la siguiente corrida.</div>
{% endif %}

<div class="meta" style="margin-top:50px;">Tiempo de ejecución del informe: {{ duracion }} s</div>

</body>
</html>
""")


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    inicio = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Ruta al CSV nuevo (preprocesado, con target)")
    parser.add_argument("--modelo", required=True,
                        help="Ruta al .pkl del modelo entrenado")
    parser.add_argument("--target", default="satisfaccion",
                        help="Nombre de la columna objetivo")
    args = parser.parse_args()

    timestamp = inicio.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_archivo = inicio.strftime("%Y%m%d_%H%M%S")

    print(f"[{timestamp}] Cargando modelo y datos...")
    modelo = cargar_modelo(args.modelo)
    df = pd.read_csv(args.csv)
    X, y_true, y_pred, y_proba = predecir(modelo, df, args.target)

    print("Calculando métricas...")
    metricas = calcular_metricas(y_true, y_pred, y_proba)

    print("Generando gráficas...")
    g_matriz = grafica_matriz_confusion(y_true, y_pred)
    g_roc = grafica_roc(y_true, y_proba, metricas["auc_roc"])
    g_pr = grafica_precision_recall(y_true, y_proba)
    g_dist = grafica_distribucion_proba(y_true, y_proba)
    g_lift = grafica_lift_chart(y_true, y_proba)
    g_gains = grafica_cumulative_gains(y_true, y_proba)

    print("Calculando SHAP (puede tardar unos segundos)...")
    shap_bar, shap_beeswarm = graficas_shap(modelo, X)

    print("Actualizando histórico...")
    historico = actualizar_historico(metricas, timestamp, len(df))
    g_historico = grafica_historico(historico)

    duracion = round((datetime.now() - inicio).total_seconds(), 2)

    html = PLANTILLA.render(
        timestamp=timestamp, nombre_csv=Path(
            args.csv).name, nombre_modelo=Path(args.modelo).name,
        n_registros=len(df), metricas=metricas,
        g_matriz=g_matriz, g_roc=g_roc, g_pr=g_pr, g_dist=g_dist,
        g_lift=g_lift, g_gains=g_gains, g_historico=g_historico, shap_bar=shap_bar,
        shap_beeswarm=shap_beeswarm, duracion=duracion,
    )

    salida = SALIDA_DIR / f"reporte_{timestamp_archivo}.html"
    salida.write_text(html, encoding="utf-8")
    print(f"✅ Informe generado en: {salida} ({duracion}s)")


if __name__ == "__main__":
    main()
