# Usa una imagen base de Python oficial y ligera
FROM python:3.11-slim

# Define el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los requerimientos e instala todas las librerías (mlflow, dvc, pandas, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el contenido de tu proyecto local
COPY . .

# Expone el puerto por defecto de MLflow
EXPOSE 5000

# El comando que se ejecuta automáticamente al iniciar el contenedor.
CMD ["mlflow", "ui", "--host", "0.0.0.0"]