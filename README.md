# Predicción Estacional de Precios Mayoristas Hortofrutícolas en Chile

## Integrantes
* Nahuel Catrileo
* Francisco Ceballos
* Angel Rocha
* Francisco Tropa

---

## 1. Definición del Problema

El sector agrícola y comercializador en Chile se enfrenta a una alta volatilidad de precios en los mercados mayoristas. Esta variación está impulsada principalmente por dinámicas geográficas, fluctuaciones en la oferta diaria (Volumen) y, fundamentalmente, por factores climáticos estacionales. Para los agricultores, intermediarios y tomadores de decisiones, la incertidumbre en el valor de cierre de las transacciones dificulta la planificación financiera a mediano plazo, la gestión de inventarios y la optimización de las cadenas de distribución.

### Pregunta de Investigación:
> ¿Es posible predecir con precisión el precio promedio por kilogramo de frutas y hortalizas en Chile utilizando algoritmos de aprendizaje automático clásico y técnicas de ingeniería de características?

---

## 2. El Dataset

El proyecto unifica y procesa los registros históricos oficiales de la Oficina de Estudios y Políticas Agrarias (ODEPA) correspondientes a los años 2025 y 2026.

* **Origen de los datos:** Plataforma de Datos Abiertos Gubernamentales de Chile.
* **Enlace oficial:** [Precios Mayoristas de Frutas y Hortalizas - datos.gob.cl](https://datos.gob.cl/dataset/precios-mayoristas-de-frutas-y-hortalizas1)

### Estructura de Directorios de Datos
Los datos crudos se unifican, limpian y dividen de forma estructurada en el subdirectorio dataset/processed/:
* **dataset/processed/dataset_unificado.csv:** Archivo conjunto de los datos 2025 y 2026 preprocesados con precios por kilo y estación.
* **dataset/processed/dataset_entrenamiento.csv:** Subconjunto de entrenamiento (72.5% del total).
* **dataset/processed/dataset_validacion.csv:** Subconjunto de validación (13.7% del total).
* **dataset/processed/dataset_prueba.csv:** Subconjunto de prueba final (13.8% del total).

---

## 3. Estrategia de Modelamiento y Pipeline de Datos

Para lograr un modelo de predicción robusto, comparable y de calidad, se ha diseñado la siguiente estrategia:

### A. Definición del Target: Precio por Kilogramo (La mejor unidad de venta)
El dataset original contiene precios bajo distintas unidades en la columna "Unidad de comercializacion" (por ejemplo, $/caja 12 kilos, $/saco 25 kilos, $/unidad). 
* **Decisión:** El target del modelo se estandariza a Precio por Kilogramo para hacer comparables todos los productos.
* **Implementación:** Se filtran las filas cuyas unidades no estén basadas en peso (removiendo unidades individuales como $/unidad o $/docena de matas que corresponden a solo el 28% del dataset, manteniendo más de 198,000 registros conjuntos entre 2025 y 2026). Extraeremos el peso numérico de las cadenas de texto (ej. de $/caja 15 kilos extraemos 15.0) y calcularemos el target como:
    Precio_Promedio_Por_Kilo = Precio promedio / Peso en Kilos

### B. Tratamiento de Variables Categóricas
* **Valores "Sin especificar":** En columnas como Variedad / Tipo y Calidad, los valores "Sin especificar" son extremadamente comunes y no deben eliminarse ni tratarse como valores perdidos (NaN). Representan una categoría válida de volumen genérico de comercio, por lo que se mantendrán como una etiqueta categórica única. Si existen valores nulos reales (NaN), se imputarán como "Sin especificar".
* **Codificación según Cardinalidad:**
    *   Baja Cardinalidad (Subsector, Estacion): Codificación One-Hot Encoding (genera columnas binarias).
    *   Alta Cardinalidad (Producto, Variedad / Tipo, Origen, Mercado, Calidad): Codificación Target Encoding (Mean Encoding) con suavizado (smoothing) y fallback al promedio global para manejar clases desconocidas que aparecen en el año 2026.

### C. Partición Cronológica de Datos (Data Splitting)
Para evitar fugas de datos temporales (temporal data leakage) donde información futura influye en predicciones pasadas, se utiliza una partición cronológica secuencial de los años 2025 y 2026:
* **Entrenamiento (72.5%):** Todo el año 2025 (desde el 2 de enero al 31 de diciembre de 2025).
* **Validación (13.7%):** El principio del año 2026 (desde el 2 de enero al 17 de marzo de 2026).
* **Prueba (13.8%):** El restante del año 2026 (desde el 17 de marzo al 29 de mayo de 2026).

---

## 4. Variables del Modelo

### Variable Objetivo (Target)
* **Precio_Promedio_Por_Kilo (Numérica):** Representa el precio promedio en pesos chilenos ($) por cada kilogramo de producto transado.

### Variables de Entrada (Features)
* **Volumen_Limpio (Numérica):** Cantidad física del producto ingresado al mercado mayorista (captura la ley de oferta y demanda).
* **Estacion (Categórica - Baja Cardinalidad):** Representa la estación del año en el Hemisferio Sur (Verano, Otoño, Invierno, Primavera) derivada del mes de la fecha.
* **Dia_Semana (Numérica):** Día de la semana de la transacción (0 a 6), permitiendo modelar fluctuaciones intradía semanales.
* **Mes (Numérica):** Componente numérico del mes (1 al 12) para modelar la estacionalidad estricta.
* **weight_kg (Numérica):** Peso en kilogramos extraído de la unidad de comercialización.
* **Subsector (Categórica - Baja Cardinalidad):** Sector de clasificación general del cultivo (Frutas, Hortalizas y tubérculos).
* **Calidad (Categórica - Cardinalidad Moderada):** Clasificación del estado o selección física del producto (e.g., Primera, Segunda, Extra).
* **Mercado (Categórica - Cardinalidad Moderada):** Mercado mayorista de origen de la transacción.
* **Producto (Categórica - Alta Cardinalidad):** Nombre específico del producto agrícola.
* **Variedad / Tipo (Categórica - Alta Cardinalidad):** Tipo específico o cultivar del producto (incluyendo "Sin especificar").
* **Origen (Categórica - Alta Cardinalidad):** Zona geográfica de origen del producto (región o provincia).

---

## 5. Modelos de Machine Learning Evaluados

El problema está abordado formalmente como una Regresión Supervisada. Se entrenan y comparan las siguientes arquitecturas:

### A. Modelos Lineales de Línea Base (Baselines)
* **Regresión Ridge / Lasso:** Modelos lineales regularizados. Ridge (L2) es el preferido por sobre Lasso (L1) para este dataset, ya que maneja de mejor manera la correlación de variables geográficas y la representación densa del Target Encoding sin excluir coeficientes de forma arbitraria.

### B. Modelos de Ensamble y Árboles
* **Random Forest Regressor:** Ensamble de tipo Bagging que promedia árboles de decisión independientes. Captura interacciones no lineales robustas.
* **Gradient Boosting Regressor:** Ensamble secuencial tipo Boosting que optimiza residuos a través de descenso de gradiente para lograr la máxima precisión en datos tabulares complejos.

---

## 6. Dockerización y Ejecución con Docker Compose

El proyecto está completamente dockerizado para facilitar su portabilidad y asegurar que el entorno de ejecución sea reproducible.

### Prerrequisitos
* **Docker** instalado en el sistema.
* **Docker Compose** instalado.

### Instrucciones de Ejecución

Para iniciar y ejecutar el contenedor del proyecto, siga los siguientes pasos desde el directorio raíz del proyecto:

1. **Construir e iniciar el contenedor:**
   ```bash
   docker compose up --build
   ```
   Este comando compilará la imagen de Docker usando el `Dockerfile` y lanzará los servicios descritos en `docker-compose.yml`.

2. **Acceder a la interfaz:**
   Una vez que el contenedor esté en ejecución, abra su navegador web e ingrese a:
   [http://localhost:8888](http://localhost:8888)
   Desde esta interfaz podrá visualizar, editar y ejecutar todos los notebooks interactivos de entrenamiento y comparación.

3. **Detener los servicios:**
   Para apagar el contenedor de manera limpia y liberar los recursos, ejecute:
   ```bash
   docker compose down
   ```
