 Definición del problema de negocio
La satisfacción del cliente es un factor estratégico clave en plataformas de comercio electrónico, ya que influye directamente en la retención de usuarios, la reputación de la empresa y los ingresos futuros. En el caso de Olist, una plataforma brasileña de e-commerce con aproximadamente 100,000 pedidos registrados entre 2016 y 2018, se observa que cerca del 23% de los clientes presentan niveles bajos de satisfacción (review_score entre 1 y 3), principalmente asociados a retrasos en las entregas y discrepancias entre la fecha prometida y la fecha real de recepción.
La insatisfacción del cliente representa un riesgo importante para el negocio debido a tres impactos principales:
Pérdida económica: clientes insatisfechos reducen la probabilidad de recompra y aumentan los costos operativos relacionados con soporte, devoluciones y compensaciones.
Pérdida de reputación: las reseñas negativas afectan la percepción de la plataforma y disminuyen la confianza de futuros compradores.
Pérdida de clientes: existe una mayor probabilidad de abandono de la plataforma por parte de usuarios con experiencias negativas.
Ante esta problemática, la analítica predictiva y los modelos de Machine Learning permiten anticipar el nivel de satisfacción del cliente antes de que este emita una reseña, utilizando variables relacionadas con logística, tiempos de entrega, pagos, vendedores y características del pedido.
Por ello, el problema de negocio se define como:
“Desarrollar un modelo predictivo capaz de anticipar el nivel de satisfacción del cliente en la plataforma Olist, antes de que el usuario publique su reseña, con el fin de permitir intervenciones tempranas que reduzcan la insatisfacción, mejoren la experiencia del cliente y contribuyan a los objetivos estratégicos de retención, reputación y eficiencia operativa”.
Este problema se alinea con los objetivos estratégicos del negocio relacionados con la mejora de la experiencia del cliente, la optimización logística y la reducción de la pérdida de usuarios mediante decisiones basadas en datos.

Hipótesis del Equipo
Hipótesis del Proyecto
Las siguientes hipótesis plantean relaciones esperadas entre variables del dataset Olist y serán evaluadas mediante análisis exploratorio de datos (EDA), análisis correlacional y modelado predictivo.
H1: Retraso en la entrega y satisfacción del cliente
Los pedidos cuya fecha real de entrega supera la fecha estimada presentan menores valores de review_score.
Relación esperada
A mayor retraso en la entrega, menor satisfacción del cliente.
Variables involucradas
order_estimated_delivery_date
order_delivered_customer_date
review_score
Justificación técnica
El tiempo de entrega es una de las variables más importantes en plataformas de e-commerce, debido a su impacto directo en la experiencia del cliente. Se espera encontrar una correlación negativa entre los días de retraso y el nivel de satisfacción.

H2: Cantidad de productos y satisfacción del cliente
Los pedidos con mayor cantidad de productos presentan menores niveles promedio de satisfacción.
Relación esperada
A mayor cantidad de productos por pedido, menor review_score promedio.
Variables involucradas
order_item_id
review_score
Justificación técnica
Pedidos con múltiples productos incrementan la complejidad logística, aumentando la probabilidad de errores en empaque, tiempos de entrega y calidad percibida del servicio.

H3: Valor del pedido y satisfacción del cliente
El valor total del pedido influye significativamente en el nivel de satisfacción del cliente.
Relación esperada
Los pedidos de mayor valor presentan diferencias significativas en review_score respecto a pedidos de menor valor.
Variables involucradas
payment_value
price
review_score
Justificación técnica
El valor económico del pedido modifica las expectativas del cliente respecto al servicio recibido. Se espera identificar patrones diferenciados de satisfacción según el monto de compra.
Seleccionar variables relevantes del dataset Olist. -> 60 metadatos -> Master Table 

Generar las bases (features) que deben ser ejecutadas cada mes.-> Automatizar el proceso • Realizar EDA exploratorio (tendencias, correlaciones, outliers, missing values). 

Métricas técnicas y de negocio iniciales (baseline)
Para el análisis exploratorio inicial (Sprint 1), se priorizarán los siguientes KPIs y métricas de negocio debido a su relación directa con el problema de satisfacción del cliente y las hipótesis planteadas:
Customer Satisfaction Rate: porcentaje de clientes satisfechos.
Unsatisfied Customer Rate: porcentaje de clientes insatisfechos.
 Average Review Score: promedio general de calificaciones de los clientes.
Late Delivery Rate: porcentaje de pedidos entregados fuera del tiempo estimado.
Average Delivery Delay: promedio de días de retraso en las entregas.
Revenue at Risk: ingresos asociados a pedidos con baja satisfacción.

De manera complementaria, se analizarán métricas segmentadas para identificar patrones específicos relacionados con la satisfacción del cliente:
Satisfaction by State: satisfacción promedio según estado o región.
Satisfaction by Payment Type: satisfacción promedio según método de pago.
Satisfaction by Product Quantity: satisfacción promedio según cantidad de productos por pedido.
Average Order Value (AOV): valor promedio de compra por pedido.
Estos indicadores permitirán validar las hipótesis planteadas, identificar variables relevantes para el modelo predictivo y establecer una línea base del comportamiento de satisfacción de los clientes en la plataforma Olist.
Nro
KPI / Métrica 
Fórmula de medición 
Variables utilizadas
Que mide
1
Customer Satisfaction Rate 
(Pedidos con review_score ≥ 4 / Total de pedidos) × 100 
review_score, order_id 
Porcentaje de clientes satisfechos  
2
Unsatisfied Customer Rate 
(Pedidos con review_score ≤ 3 / Total de pedidos) × 100
review_score, order_id 
Porcentaje de clientes insatisfechos 
3
Average Review Score 
Promedio(review_score) 
review_score  
Nivel promedio de satisfacción 
4
Late Delivery Rate 
(Pedidos entregados tarde / Total de pedidos) × 100 
order_delivered_customer_date, order_estimated_delivery_date 
Porcentaje de pedidos entregados fuera del tiempo estimado 
5
Average Delivery Delay 
Promedio(fecha_entrega_real − fecha_estimada) 
order_delivered_customer_date, order_estimated_delivery_date 
Promedio de dias de retraso entre las entregas. 
6
Revenue at Risk 
SUM(payment_value) donde review_score ≤ 3 
payment_value, review_score 
Ingresos asociados a clientes insatisfechos.
7
Average Order Value (AOV) 
SUM(payment_value) / Total de pedidos 
payment_value, order_id 
Valor promedio de compra por pedido 
8
Satisfaction by State 
Promedio(review_score) por estado 
customer_state, review_score 
Nivel de satisfacción según ubicación geográfica 
9
Satisfaction by Payment Type 
Promedio(review_score) por método de pago 
payment_type, review_score 
Relación entre método de pago y satisfacción 
10
Satisfaction by Product Quantity 
Promedio(review_score) según cantidad de productos por pedido 
order_item_id, review_score 
Relación entre complejidad del pedido y satisfacción 



Fuentes para tu informe IEEE:
IEEE Xplore — Machine Learning Approach to Predict E-commerce Customer Satisfaction Score → https://ieeexplore.ieee.org/document/10147542/
Towards Data Science — Customer Satisfaction Prediction Using Machine Learning → https://towardsdatascience.com/customer-satisfaction-prediction-using-machine-learning
ResearchGate — Enhancing Customer Satisfaction using Machine Learning → https://www.researchgate.net/publication/385738707
Medium — Case Study: Olist Customer Satisfaction Prediction → https://rushikeshdarge.medium.com/olist-customer-satisfaction-prediction
Zendesk — Machine Learning para experiencia del cliente → https://www.zendesk.com.mx/blog


