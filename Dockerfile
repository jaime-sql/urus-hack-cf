FROM python:3.10-slim

# Evitar la creación de archivos .pyc y forzar logs sin buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias primero para optimizar el caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

EXPOSE 8000

# Iniciar Chainlit
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
