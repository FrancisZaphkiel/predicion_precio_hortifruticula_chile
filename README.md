# Predicción Estacional de Precios Mayoristas Hortofrutícolas en Chile

## 1. Definición del Problema

El sector agrícola y comercializador en Chile se enfrenta a una alta volatilidad de precios en los mercados mayoristas. Esta variación está impulsada principalmente por dinámicas geográficas, fluctuaciones en la oferta diaria (`Volumen`) y, fundamentalmente, por factores climáticos estacionales. Para los agricultores, intermediarios y tomadores de decisiones, la incertidumbre en el valor de cierre de las transacciones dificulta la planificación financiera a mediano plazo, la gestión de inventarios y la optimización de las cadenas de distribución.

**Pregunta de Investigación:**

> ¿Es posible predecir con precisión el **Precio Promedio estacional** de frutas y hortalizas en Chile utilizando algoritmos de aprendizaje automático clásico y técnicas de ingeniería de características?

---

## Integrantes

* Nahuel Catrileo
* Francisco Ceballos
* Angel Rocha
* Francisco Tropa

---

## 2. El Dataset

El proyecto unifica y procesa los registros históricos oficiales de la **Oficina de Estudios y Políticas Agrarias (ODEPA)** correspondientes a los años **2025 y 2026** (con más de 270.000 registros entre los dos años).

* **Origen de los datos:** Plataforma de Datos Abiertos Gubernamentales de Chile.
* **Enlace oficial:** [Precios Mayoristas de Frutas y Hortalizas - datos.gob.cl](https://datos.gob.cl/dataset/precios-mayoristas-de-frutas-y-hortalizas1)

### Arquitectura de Variables (Features)

* **Temporales:** `Fecha` (Registro cronológico continuo).
* **Geográficas / Logísticas:** `Región`, `Mercado`, `Origen` (Procedencia a nivel provincial o internacional).
* **Segmentación de Producto:** `Subsector` (Frutas o Hortalizas), `Producto`, `Variedad / Tipo`, `Calidad`.
* **Métricas Operacionales:** `Volumen` (Cantidad de unidades o envases transados).
* **Variable Objetivo ($Y$):** `Precio promedio` (Valor continuo).

---

## 3. Objetivo de la Predicción Estacional

A diferencia de un análisis predictivo diario inmediato, este modelo implementa una **estrategia macro-temporal**:

1. **Aislamiento del Ciclo Biológico:** A partir de la columna `Fecha`, se realiza una ingeniería de variables para extraer la característica categórica **`Estacion_Del_Año`** (*Verano, Otoño, Invierno, Primavera*). Esto permite al modelo mapear cómo cambia la sensibilidad de la ley de oferta y demanda según la época climática de cosecha.
2. **Utilidad del Negocio:** Habilita la capacidad de predecir el comportamiento del precio para el próximo ciclo completo (por ejemplo, proyectar en invierno el precio promedio esperado para la producción que ingresará en la primavera del año siguiente), optimizando las decisiones estratégicas de siembra, rotación de cultivos y negociación de contratos de distribución.

---

## 4. Modelos y Técnicas de Machine Learning

El problema está abordado formalmente como una **Regresión Supervisada**. Tras un riguroso proceso de limpieza de datos (remoción de registros inactivos con precios en cero, conversión de formatos de texto `"0,0000"` a flotantes numéricos y el uso de **Modelos de Lenguaje (LLM) institucionales** para homogeneizar la columna `Unidad de comercializacion`), se entrenan y comparan las siguientes arquitecturas:

### A. Línea Base (Baseline)

* **Regresión Ridge y Lasso (Modelos Lineales Regularizados):**
* **Técnica:** Modelado lineal con penalizaciones estadísticas sobre los coeficientes ($L_2$ para Ridge, $L_1$ para Lasso).
* **Propósito:** Evaluar si el precio estacional se puede aproximar mediante una combinación lineal simple, controlando el sobreajuste (*overfitting*) provocado por la alta cardinalidad de los productos y orígenes.



### B. Modelos Avanzados basados en Árboles (Ensambles)

* **Random Forest Regressor:**
* **Técnica:** Ensamble mediante *Bagging* (entrenamiento en paralelo de múltiples árboles de decisión independientes con subconjuntos aleatorios de datos).
* **Propósito:** Capturar interacciones no lineales complejas entre variables categóricas (como el cruce de un `Producto` con una `Calidad` específica en una determinada `Estacion_Del_Año`) sin requerir un escalado estricto de las métricas de volumen.


* **Gradient Boosting Regressor (o XGBoost):**
* **Técnica:** Ensamble mediante *Boosting* (construcción secuencial y optimizada de árboles de decisión, donde cada árbol corrige matemáticamente los errores residuales del anterior utilizando el descenso de gradiente).
* **Propósito:** Maximizar la precisión en datos tabulares complejos, capturando de manera óptima las sutiles fluctuaciones operativas del mercado.



---

## 5. Estrategias de Preprocesamiento Conectadas

Para que los algoritmos procesen eficientemente las categorías de texto, la ingeniería de características se divide según el **Análisis de Cardinalidad**:

* **One-Hot Encoding (Baja Cardinalidad):** Aplicado a `Estacion_Del_Año`, `Subsector` y `Calidad`. Crea columnas binarias independientes preservando la interpretabilidad pura.
* **Target Encoding (Alta Cardinalidad):** Aplicado a `Producto`, `Variedad / Tipo` y `Origen`. Reemplaza las categorías de texto masivas por el promedio del precio histórico correspondiente, evitando la inflación artificial de columnas y reduciendo significativamente los tiempos de cómputo.
