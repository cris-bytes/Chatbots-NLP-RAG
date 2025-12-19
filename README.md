# AI Chatbot con RAG (Retrieval Augmented Generation)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Asistente conversacional inteligente basado en IA** que responde preguntas sobre documentación técnica usando técnicas modernas de NLP, embeddings vectoriales y Large Language Models (LLMs).

---

## Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Características Principales](#-características-principales)
- [Arquitectura](#-arquitectura)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Documento Técnico Seleccionado](#-documento-técnico-seleccionado)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [API Documentation](#-api-documentation)
- [Ejemplos de Uso](#-ejemplos-de-uso)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

---

## Descripción del Proyecto

Este proyecto implementa un **chatbot RAG (Retrieval Augmented Generation)** completo que:

1. **Ingesta documentos técnicos** (PDF, URL, texto)
2. **Procesa y vectoriza** el contenido usando embeddings semánticos
3. **Almacena en una base vectorial** (FAISS) para búsqueda eficiente
4. **Responde preguntas** combinando búsqueda vectorial + LLM (GPT-3.5)
5. **Expone una API REST** para integración con múltiples canales

### Técnicas Implementadas

- - **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- - **Vector Store**: FAISS (Facebook AI Similarity Search)
- - **RAG**: Retrieval Augmented Generation
- - **LLM**: OpenAI GPT-3.5-turbo
- - **Prompt Engineering**: Control de alucinaciones
- - **Recuperación Contextual**: Top-K similarity search
- - **API REST**: FastAPI con documentación automática

---

## Características Principales

### Ingesta Inteligente de Documentos

- Soporte para múltiples formatos (PDF, HTML, texto)
- Limpieza y normalización automática
- Chunking con overlap para mantener contexto
- Metadata tracking

### Búsqueda Vectorial Eficiente

- Embeddings de 384 dimensiones
- Similitud coseno con FAISS
- Búsqueda en milisegundos
- Persistencia en disco

### RAG Avanzado

- Control de alucinaciones con threshold de relevancia
- Confidence scoring
- Fallback inteligente cuando no hay información
- Context quality assessment

### API REST Moderna

- FastAPI con validación automática
- Documentación interactiva (Swagger/ReDoc)
- CORS habilitado
- Type safety con Pydantic

### Conversación Multi-Turn (Básico)

- Cada pregunta es independiente
- Preparado para extensión con memoria conversacional
- Session management ready

---

## Arquitectura

```
┌────────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION                      │
│  PDF/URL/Text → Extract → Clean → Chunk → Embed → Index   │
└────────────────────────────────────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│                       RAG PIPELINE                         │
│  Question → Embed → Search → Assess → Prompt → LLM → Response │
└────────────────────────────────────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│                        API LAYER                           │
│         FastAPI (POST /ask, GET /health, GET /stats)       │
└────────────────────────────────────────────────────────────┘
```

Para más detalles, ver [docs/architecture.md](docs/architecture.md).

---

## Requisitos

### Software

- **Python**: 3.9 o superior
- **pip**: Gestor de paquetes de Python
- **Git**: Para clonar el repositorio

### Dependencias Principales

- `fastapi` - Framework web moderno
- `uvicorn` - Servidor ASGI
- `sentence-transformers` - Modelos de embeddings
- `faiss-cpu` - Vector store
- `PyPDF2` - Procesamiento de PDFs
- `openai` - Integración con OpenAI API
- `pydantic` - Validación de datos

Ver [requirements.txt](requirements.txt) para la lista completa.

### API Keys (Opcional)

- **OpenAI API Key**: Para usar GPT-3.5-turbo
  - Obtener en: https://platform.openai.com/api-keys
  - Sin API key, el sistema funciona en modo degradado

---

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/Chatbots-NLP-RAG.git
cd Chatbots-NLP-RAG
```

### 2. Crear Entorno Virtual

```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Nota**: La primera vez descargará el modelo de embeddings (~90MB), puede tardar unos minutos.

### 4. Configurar Variables de Entorno

```bash
cp .env.example .env
```

Editar `.env` y agregar tu API key de OpenAI (opcional):

```env
OPENAI_API_KEY=sk-your-api-key-here
```

---

## Uso Rápido

### 1. Ingestar Documento

El proyecto incluye un documento de ejemplo sobre FastAPI en `data/raw/fastapi_guide.txt`.

```bash
python main.py --ingest data/raw/fastapi_guide.txt
```

**Salida esperada**:
```
Processing document: data/raw/fastapi_guide.txt
Extracting text from text...
Extracted 15234 characters
Cleaning text...
Cleaned text: 14892 characters
Creating chunks...
Created 28 chunks

Generating embeddings...
100%|████████████████████| 28/28 [00:05<00:00,  5.2it/s]

Creating vector store...
Created FAISS index with dimension 384
Added 28 vectors to index. Total: 28

Saving vector store...
Saved vector store to ./data/vectorstore

INGESTION COMPLETE!
```

### 2. Modo Interactivo (Q&A en Terminal)

```bash
python main.py --interactive
```

**Ejemplo de sesión**:
```
You: What is FastAPI?

Bot: FastAPI is a modern, fast web framework for building APIs with Python 3.7+
based on standard Python type hints. It provides high performance comparable to
NodeJS and Go, automatic API documentation, and built-in data validation.

Confidence: 87%
Sufficient context: True

Sources:
  1. Score: 0.856 - FastAPI is a modern, fast (high-performance), web framework...
```

### 3. Iniciar API Server

```bash
python main.py --api
```

**Salida**:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Acceder a:
- API Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

### 4. Hacer Preguntas via API

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I install FastAPI?"}'
```

**Respuesta**:
```json
{
  "answer": "To install FastAPI, you need Python 3.7 or higher. Use pip to install FastAPI and an ASGI server like Uvicorn with the command: pip install fastapi and pip install uvicorn[standard]",
  "context": ["..."],
  "sources": [...],
  "confidence": 0.82,
  "has_sufficient_context": true
}
```

---

## Documento Técnico Seleccionado

### Documento: FastAPI - Modern Web Framework Guide

**Ubicación**: `data/raw/fastapi_guide.txt`

### ¿Por qué este documento?

1. **Contenido Técnico Rico**: Cubre conceptos, APIs, ejemplos de código, best practices
2. **Bien Estructurado**: Secciones claras, jerarquía lógica
3. **Tamaño Apropiado**: ~15K caracteres, suficiente para demostrar RAG
4. **Multidominio**: Instalación, configuración, desarrollo, deployment
5. **Relevante**: FastAPI es el framework usado en este proyecto

### Tipos de Preguntas que puede Responder

- **Instalación**: "How do I install FastAPI?"
- **Configuración**: "What are FastAPI's key features?"
- **Desarrollo**: "How do I create a FastAPI endpoint?"
- **Validación**: "How does FastAPI handle request validation?"
- **Seguridad**: "What security features does FastAPI offer?"
- **Deployment**: "How can I deploy FastAPI to production?"
- **Best Practices**: "What are FastAPI best practices?"

### Retos del Texto

1. **Código embebido**: Ejemplos de Python mezclados con texto
2. **Comandos**: Comandos de terminal y pip
3. **Listas**: Muchas listas de características
4. **Términos técnicos**: Vocabulario específico de web frameworks

**Solución Implementada**: El chunking basado en sentencias mantiene el contexto de código y texto juntos. La limpieza preserva caracteres especiales relevantes.

---

## 📁 Estructura del Proyecto

```
Chatbots-NLP-RAG/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── document_processor.py   # PARTE 1: Document ingestion
│   ├── embeddings.py            # Embedding generation
│   ├── vector_store.py          # FAISS vector store
│   ├── rag_engine.py            # PARTE 2: RAG pipeline
│   └── api.py                   # PARTE 3: REST API
├── data/
│   ├── raw/                     # Original documents
│   │   └── fastapi_guide.txt   # Sample document
│   ├── processed/               # Processed chunks
│   │   └── chunks.json         # Serialized chunks
│   └── vectorstore/             # FAISS index
│       ├── faiss_index.bin     # Vector index
│       ├── chunks.pkl          # Chunk metadata
│       └── metadata.pkl        # Index metadata
├── docs/
│   └── architecture.md          # PARTE 4: Architecture doc
├── tests/
│   └── test_rag.py             # Unit tests
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file (PARTE 5)
```

---

## API Documentation

### Endpoints

#### `POST /ask`

Hacer una pregunta al chatbot.

**Request**:
```json
{
  "question": "What is FastAPI?",
  "top_k": 3  // Optional, default: 3
}
```

**Response**:
```json
{
  "answer": "FastAPI is a modern web framework...",
  "context": ["chunk1", "chunk2", "chunk3"],
  "sources": [
    {
      "chunk_id": 0,
      "source": "data/raw/fastapi_guide.txt",
      "score": 0.85,
      "preview": "FastAPI is a modern..."
    }
  ],
  "confidence": 0.85,
  "has_sufficient_context": true
}
```

#### `GET /health`

Verificar estado del sistema.

**Response**:
```json
{
  "status": "healthy",
  "message": "API is running",
  "vector_store_loaded": true
}
```

#### `GET /stats`

Obtener estadísticas del vector store.

**Response**:
```json
{
  "num_vectors": 28,
  "num_chunks": 28,
  "embedding_dim": 384,
  "is_trained": true
}
```

### Interactive Documentation

Una vez iniciado el servidor, acceder a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Ejemplos de Uso

### Ejemplo 1: Ingestar tu Propio Documento

```bash
# PDF
python main.py --ingest /path/to/document.pdf --type pdf

# URL (web scraping)
python main.py --ingest https://example.com/docs --type url

# Texto plano
python main.py --ingest /path/to/document.txt --type text

# Auto-detect
python main.py --ingest /path/to/document.pdf
```

### Ejemplo 2: Python Client

```python
import requests

def ask_question(question: str):
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": question}
    )
    return response.json()

# Usar
result = ask_question("What is FastAPI?")
print(result['answer'])
```

### Ejemplo 3: JavaScript/Web Integration

```javascript
async function askChatbot(question) {
  const response = await fetch('http://localhost:8000/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question})
  });

  const data = await response.json();
  console.log(data.answer);
}

askChatbot("How do I create a FastAPI endpoint?");
```

---

## Testing

### Ejecutar Tests

```bash
# Instalar pytest si no está
pip install pytest

# Ejecutar todos los tests
pytest

# Con verbose
pytest -v

# Con coverage
pytest --cov=src tests/
```

### Crear Tests Personalizados

```python
# tests/test_custom.py
from src.rag_engine import RAGEngine

def test_answer_quality():
    engine = RAGEngine()
    engine.load_vector_store()

    result = engine.answer_question("What is FastAPI?")

    assert result['has_sufficient_context'] == True
    assert result['confidence'] > 0.5
    assert len(result['answer']) > 0
```

---

## Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build y Run**:
```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx rag-chatbot
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
```

**Run**:
```bash
docker-compose up
```

### Cloud Deployment

#### AWS

```bash
# Elastic Beanstalk
eb init -p python-3.9 rag-chatbot
eb create rag-chatbot-env
eb deploy
```

#### Google Cloud Platform

```bash
# Cloud Run
gcloud run deploy rag-chatbot \
  --source . \
  --platform managed \
  --region us-central1
```

#### Heroku

```bash
heroku create rag-chatbot
git push heroku main
```

---

## Configuración Avanzada

### Cambiar Modelo de Embeddings

Editar `src/config.py` o `.env`:

```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

Modelos recomendados:
- `all-MiniLM-L6-v2` (rápido, 384-dim)
- `all-mpnet-base-v2` (balance, 768-dim)
- `multi-qa-mpnet-base-dot-v1` (Q&A optimizado)

### Cambiar LLM

```env
# OpenAI
LLM_MODEL=gpt-4

# O usar modelo local (requiere instalación adicional)
LLM_MODEL=local:llama-2-7b
```

### Ajustar Parámetros RAG

```env
TOP_K_CHUNKS=5        # Más contexto
CHUNK_SIZE=1000       # Chunks más grandes
CHUNK_OVERLAP=100     # Más overlap
TEMPERATURE=0.3       # Respuestas más deterministas
```

---

## Contribución

Este proyecto es una prueba técnica, pero las contribuciones son bienvenidas.

### Áreas de Mejora

1. **Memoria Conversacional**: Implementar historial de sesiones
2. **Múltiples Modelos**: Soporte para múltiples LLMs
3. **Caché**: Redis para respuestas frecuentes
4. **Monitoreo**: Prometheus + Grafana
5. **Tests**: Aumentar cobertura de tests

### Proceso

1. Fork el proyecto
2. Crear branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Abrir Pull Request

---

## Métricas de Calidad

### Cobertura de Código

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Linting

```bash
# Black (formatting)
black src/

# Flake8 (linting)
flake8 src/
```

---

## Troubleshooting

### Error: "Vector store not found"

**Solución**: Ejecutar primero la ingesta:
```bash
python main.py --ingest data/raw/fastapi_guide.txt
```

### Error: "OpenAI API rate limit"

**Solución**: Agregar delays o usar plan pago de OpenAI

### Error: "CUDA out of memory"

**Solución**: El proyecto usa CPU por defecto. Si modificaste para usar GPU, reduce batch size.

### Lentitud en Primera Ejecución

**Solución**: Normal. Está descargando el modelo de embeddings (~90MB). Siguientes ejecuciones serán rápidas.

---

## Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

---

## Autor

**Prueba Técnica - Ingeniero IA/ML**

- Proyecto: RAG Chatbot System
- Stack: Python, FastAPI, FAISS, OpenAI, Sentence-Transformers
- Fecha: 2024

---

## Agradecimientos

- [FastAPI](https://fastapi.tiangolo.com/) por el excelente framework
- [Sentence-Transformers](https://www.sbert.net/) por los modelos de embeddings
- [FAISS](https://faiss.ai/) por la búsqueda vectorial eficiente
- [OpenAI](https://openai.com/) por el API de LLMs

---

## Recursos Adicionales

- [Architecture Documentation](docs/architecture.md)
- [FastAPI Official Docs](https://fastapi.tiangolo.com/)
- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [Sentence-Transformers Docs](https://www.sbert.net/)

---

**¡Gracias por revisar este proyecto!** 🚀

Si tienes preguntas o sugerencias, no dudes en abrir un issue o contactar.
