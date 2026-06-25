FROM python:3.12-slim

# Instalar dependencias del sistema necesarias para construir algunos paquetes de Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos del proyecto
COPY . .

# Exponer el puerto por defecto de Jupyter
EXPOSE 8888

# Comando para iniciar Jupyter Notebook deshabilitando tokens para facilitar la visualización en entornos de desarrollo local
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
