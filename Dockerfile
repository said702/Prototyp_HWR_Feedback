# Basis: Python + Slim-Image
FROM python:3.9-slim

# Arbeitsverzeichnis
WORKDIR /app

# System-Abhängigkeiten
RUN apt-get update && apt-get install -y curl build-essential git && apt-get clean

# Poetry installieren
RUN curl -sSL https://install.python-poetry.org | python3 -

# Poetry in PATH setzen
ENV PATH="/root/.local/bin:$PATH"

# Poetry Konfiguration: keine virtuellen Umgebungen
ENV POETRY_VIRTUALENVS_CREATE=false

# pyproject und lockfile kopieren
COPY pyproject.toml poetry.lock* /app/

# Abhängigkeiten installieren (ohne das Projekt selbst zu installieren)
RUN poetry install  --no-root --only main

# Den Rest des Projekts kopieren
COPY . /app

# Startbefehl
CMD ["poetry", "run", "python", "Main.py"]
