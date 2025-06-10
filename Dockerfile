FROM python:3.10-slim

WORKDIR /app

# Instale dependências do sistema necessárias para o OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copie os arquivos do projeto para o container
COPY . .

# Instale as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Use Gunicorn para rodar o serviço Flask em produção
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "120", "yolo_verification_service:app"]