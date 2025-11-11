# Base de la imagen
FROM python:3.12-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar todo el proyecto
COPY . /app

# Instalar dependencias
RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer puerto 8000
EXPOSE 8000

# Etiqueta para que Docker Desktop mapee automáticamente el puerto
LABEL com.docker.desktop.port.8000="8000"

# Verificación de salud 
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s \
  CMD curl -f http://localhost:8000/health || exit 1

# Comando de arranque del servidor FastAPI
ENTRYPOINT ["uvicorn"]
CMD ["src.serving.CaravanInsuranceModelService:app", "--host", "0.0.0.0", "--port", "8000"]