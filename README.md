# Predicción Estacional de Precios Mayoristas Hortofrutícolas en Chile

## Integrantes
* Nahuel Catrileo
* Francisco Ceballos
* Angel Rocha
* Francisco Tropa

<img width="3072" height="3793" alt="Papelografo" src="https://github.com/user-attachments/assets/ba87713b-822f-49cf-a12f-29c4d350ae35" />

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

## 4. Variables del Modelo (Diccionario de Datos)

Para facilitar la comprensión técnica y comercial de los datos dentro del equipo, a continuación se describe en detalle qué representa cada una de las variables utilizadas en los modelos:

### Variable Objetivo (Target)
* **`Precio_Promedio_Por_Kilo` (Numérica):** Representa el precio promedio en pesos chilenos ($) calculado por cada kilogramo de producto transado. Se calcula como `Precio promedio / weight_kg`. Esta normalización es crítica para poder comparar el valor real del producto de forma justa, sin importar el formato de envase en el que se venda.

### Variables de Entrada (Features)
* **`Mercado` (Categórica - Cardinalidad Moderada):** El mercado mayorista físico donde se realiza la transacción comercial del producto (ej. *Lo Valledor*, *Vega Central Mapocho*, *Terminal Hortofrutícola de Chillán*). Permite modelar variaciones de precio asociadas a la ubicación, costos logísticos de transporte y condiciones de oferta/demanda locales de cada centro de distribución.
* **`Subsector` (Categórica - Baja Cardinalidad):** Sector o clasificación general del cultivo. En este conjunto de datos se divide principalmente en:
    *   *Frutas*: Frutas frescas, deshidratadas o secas.
    *   *Hortalizas y tubérculos*: Verduras frescas, legumbres y papas.
* **`Producto` (Categórica - Alta Cardinalidad):** Nombre específico o genérico de la especie agrícola transada (ej. *Manzana*, *Papa*, *Tomate*, *Limón*). Es la variable principal de agrupación del producto.
* **`Variedad / Tipo` (Categórica - Alta Cardinalidad):** Tipo específico o variedad comercial del producto (ej. Manzana *Royal Gala*, Papa *Custodia*, Tomate *Larga Vida*). La diferenciación de variedad es de alto impacto, dado que el mercado suele pagar precios muy diferentes por distintas variedades de una misma especie.
* **`Calidad` (Categórica - Cardinalidad Moderada):** Grado de selección o calidad física asignado al lote comercializado (ej. *Primera*, *Segunda*, *Tercera*). Los productos clasificados como de "Primera" calidad obtienen valores comerciales significativamente más altos en los mercados.
* **`Origen` (Categórica - Alta Cardinalidad):** Zona de procedencia geográfica (comuna, provincia o región) donde se cosechó el producto (ej. *Curicó*, *Coquimbo*, *Angol*). Ayuda a estimar los costos de flete, la reputación de la zona productora y ventanas de cosecha locales.
* **`Volumen_Limpio` (Numérica):** Cantidad total de producto ingresado al mercado en esa transacción (limpio de valores nulos o caracteres erróneos). Representa el volumen del lote comercializado, capturando la dinámica de oferta.
* **`weight_kg` (Numérica):** Peso neto equivalente en kilogramos del formato o envase de comercialización utilizado (ej. si se vende en saco de 25 kg, el valor es `25.0`; si es por kilo individual, es `1.0`). Es la constante de normalización para calcular el precio por kilo.
* **`Mes` (Numérica):** Mes calendario de la transacción (1 al 12). Captura patrones de estacionalidad histórica de precios a lo largo del año (ej. alzas en invierno y bajas en temporada de cosecha).
* **`Dia_Semana` (Numérica):** Día de la semana en formato entero (0 para Lunes, 6 para Domingo). Captura variaciones intradía asociadas a los días de mayor o menor descarga de mercadería en los mercados.
* **`Estacion` (Categórica - Baja Cardinalidad):** Estación del año en el hemisferio sur (Verano, Otoño, Invierno, Primavera) deducida a partir del mes de la fecha. Captura dinámicas climáticas generales y de ciclos agrícolas.

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
