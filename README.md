# Predicción de Precios Hortofrutícolas - ODEPA

Este proyecto tiene como objetivo diagnosticar, explorar y preparar la infraestructura para la predicción del precio promedio por kilogramo de productos agrícolas en los mercados mayoristas de Chile, utilizando los datasets históricos de la Oficina de Estudios y Políticas Agrarias (ODEPA), como 2025.csv.

---

## Configuración del Entorno de Desarrollo (Fish Shell)

Para preparar el entorno virtual de Python y ejecutar las dependencias en la terminal Fish Shell de Linux, siga los siguientes pasos de forma secuencial:

### 1. Creación del Entorno Virtual
Cree un entorno virtual de Python en la carpeta raíz del proyecto ejecutando:
```fish
python3 -m venv .venv
```

### 2. Activación del Entorno en Fish Shell
En Fish Shell, active el entorno usando la sintaxis source:
```fish
source .venv/bin/activate.fish
```

### 3. Instalación de Dependencias
Con el entorno virtual activado, instale las dependencias especificadas en el archivo requirements.txt:
```fish
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Registro del Kernel en Jupyter
Registre el entorno virtual como un kernel de Jupyter para que esté disponible en los notebooks:
```fish
python3 -m ipykernel install --user --name=prediccion-precios --display-name "Predicción Precios (ODEPA)"
```

### 5. Ejecución de Jupyter
Para iniciar el entorno de Jupyter Notebook o Jupyter Lab, ejecute:
```fish
jupyter notebook
```
o bien:
```fish
jupyter lab
```

---

## Estrategia de Modelamiento y Pipeline de Datos

Para lograr un modelo de predicción robusto, comparable y de calidad, se ha diseñado la siguiente estrategia:

### 1. Definición del Target: Precio por Kilogramo (La mejor unidad de venta)
El dataset original contiene precios bajo distintas unidades en la columna "Unidad de comercializacion" (por ejemplo, $/caja 12 kilos, $/saco 25 kilos, $/unidad). 
*   **Decisión:** El target del modelo se estandarizará a Precio por Kilogramo para hacer comparables todos los productos.
*   **Implementación:** Se filtrarán las filas cuyas unidades no estén basadas en peso (removiendo unidades individuales como $/unidad o $/docena de matas que corresponden a solo el 28% del dataset, manteniendo más de 144,000 registros). Extraeremos el peso numérico de las cadenas de texto (ej. de $/caja 15 kilos extraemos 15.0) y calcularemos el target como:
    Precio_Promedio_Por_Kilo = Precio promedio / Peso en Kilos

### 2. Variables de Entrada (Features) y Justificación
*   **Volumen:** Representa la oferta en el mercado. Según la ley de oferta y demanda, un alto volumen transado suele reducir el precio promedio, siendo una característica fundamental.
*   **Estacion (Temporal):** Reemplaza la fecha lineal por las estaciones del año en el Hemisferio Sur (Chile), capturando ciclos climáticos y de cosechas:
    *   Verano: Diciembre, Enero, Febrero.
    *   Otoño: Marzo, Abril, Mayo.
    *   Invierno: Junio, Julio, Agosto.
    *   Primavera: Septiembre, Octubre, Noviembre.
*   **Dia_Semana (Temporal):** Captura micro-patrones de negociación (por ejemplo, si los lunes y viernes los precios varían debido al abastecimiento para el fin de semana).
*   **Variables Categóricas Estructurales:** Subsector, Producto, Variedad / Tipo, Calidad, Origen, y Mercado (permiten al modelo conocer el tipo de cultivo, su calidad física, la zona geográfica de producción y el centro mayorista de venta).

### 3. Tratamiento de Variables Categóricas
*   **Valores "Sin especificar":** En columnas como Variedad / Tipo y Calidad, los valores "Sin especificar" son extremadamente comunes y no deben eliminarse ni tratarse como valores perdidos (NaN). Representan una categoría válida de volumen genérico de comercio, por lo que se mantendrán como una etiqueta categórica única. Si existen valores nulos reales (NaN), se imputarán como "Sin especificar".
*   **Codificación según Cardinalidad:**
    *   Baja Cardinalidad (Subsector, Estacion): Codificación One-Hot Encoding (genera columnas binarias).
    *   Alta Cardinalidad (Producto, Variedad / Tipo, Origen, Mercado, Calidad): Codificación Target Encoding (Mean Encoding) basado en el target de entrenamiento, para evitar expandir artificialmente la dimensionalidad y acelerar el cómputo de los modelos de ensamble.

### 4. Partición de Datos (Data Splitting)
Para simular correctamente un entorno de producción y evitar la fuga de datos (data leakage) de naturaleza temporal, no utilizaremos partición aleatoria. En su lugar, dividiremos cronológicamente los datos de 2025:
*   **Entrenamiento:** Datos de Enero a Septiembre de 2025.
*   **Validación / Pruebas:** Datos de Octubre a Diciembre de 2025.

---

## Variables del Modelo

### Variable Objetivo (Target)
*   **Precio_Promedio_Por_Kilo (Numérica):** Representa el precio promedio en pesos chilenos ($) por cada kilogramo de producto transado. Es calculada dividiendo el "Precio promedio" del registro por el peso extraído de la "Unidad de comercialización".

### Variables de Entrada (Features)
*   **Volumen (Numérica):** Cantidad física del producto ingresado al mercado mayorista. Permite capturar la ley de oferta y demanda.
*   **Estacion (Categórica - Baja Cardinalidad):** Representa la estación del año en el Hemisferio Sur (Verano, Otoño, Invierno, Primavera) derivada del mes de la fecha.
*   **Dia_Semana (Numérica/Categórica):** Día de la semana de la transacción (0 a 6), permitiendo modelar fluctuaciones intradía semanales.
*   **Subsector (Categórica - Baja Cardinalidad):** Sector de clasificación general del cultivo (e.g., Frutas, Hortalizas y tubérculos).
*   **Calidad (Categórica - Cardinalidad Moderada):** Clasificación del estado o selección física del producto (e.g., Primera, Segunda, Extra).
*   **Mercado (Categórica - Cardinalidad Moderada):** Mercado mayorista de origen de la transacción.
*   **Producto (Categórica - Alta Cardinalidad):** Nombre específico del producto agrícola (e.g., Manzana, Tomate, Ajo).
*   **Variedad / Tipo (Categórica - Alta Cardinalidad):** Tipo específico o cultivar del producto (e.g., Royal Gala, Granny Smith, Chino). Incluye la categoría "Sin especificar" para registros genéricos.
*   **Origen (Categórica - Alta Cardinalidad):** Zona geográfica de origen del producto (región o provincia).
