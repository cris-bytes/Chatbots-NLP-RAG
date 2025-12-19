# Arquitectura del Sistema RAG Chatbot

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura General](#arquitectura-general)
3. [Componentes del Sistema](#componentes-del-sistema)
4. [Flujo de Datos](#flujo-de-datos)
5. [Decisiones de Diseño](#decisiones-de-diseño)
6. [Limitaciones](#limitaciones)
7. [Escalabilidad en Producción](#escalabilidad-en-producción)
8. [Memoria Conversacional](#memoria-conversacional)
9. [Integración con Canales](#integración-con-canales)
10. [Monitoreo y Observabilidad](#monitoreo-y-observabilidad)

---

## Resumen Ejecutivo

Este sistema implementa un chatbot conversacional basado en **RAG (Retrieval Augmented Generation)** capaz de responder preguntas sobre documentación técnica. El sistema combina:

- **Embeddings semánticos** para representación vectorial de texto
- **Búsqueda vectorial** con FAISS para recuperación eficiente
- **LLMs** (Large Language Models) para generación de respuestas contextualizadas
- **API REST** para integración con múltiples canales

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Document Source  ──►  Extraction  ──►  Cleaning  ──►  Chunking │
│  (PDF/URL/Text)        (PyPDF2/BS4)     (Regex)      (Overlap)  │
│                                                                  │
│                            ▼                                     │
│                                                                  │
│                    Embedding Generation                          │
│                    (Sentence-Transformers)                       │
│                                                                  │
│                            ▼                                     │
│                                                                  │
│                      Vector Indexing                             │
│                         (FAISS)                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Query  ──►  Query Embedding  ──►  Vector Search           │
│                   (Same model)          (FAISS)                  │
│                                                                  │
│                            ▼                                     │
│                                                                  │
│                   Top-K Context Retrieval                        │
│                   (Similarity scores)                            │
│                                                                  │
│                            ▼                                     │
│                                                                  │
│                   Context Quality Assessment                     │
│                   (Relevance threshold)                          │
│                                                                  │
│                            ▼                                     │
│                                                                  │
│              Prompt Construction + LLM Call                      │
│              (OpenAI GPT / Local Model)                          │
│                                                                  │
│                            ▼                                     │
│                                                                  │
│                  Response + Sources + Confidence                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          API LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FastAPI Server                                                  │
│  ├── POST /ask      (Question answering)                        │
│  ├── GET  /health   (Health check)                              │
│  └── GET  /stats    (Vector store stats)                        │
│                                                                  │
│  CORS enabled for web clients                                   │
│  Automatic OpenAPI documentation                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Componentes del Sistema

### 1. Document Processor (`document_processor.py`)

**Responsabilidad**: Ingesta y procesamiento de documentos

**Funcionalidades**:
- Extracción de texto desde múltiples fuentes (PDF, URL, texto plano)
- Limpieza y normalización de texto
- Chunking inteligente con overlap para mantener contexto
- Serialización de chunks procesados

**Técnicas utilizadas**:
- PyPDF2 para extracción de PDFs
- BeautifulSoup para web scraping
- Regex para limpieza de texto
- Chunking basado en sentencias para mejores límites semánticos

### 2. Embedding Generator (`embeddings.py`)

**Responsabilidad**: Generación de representaciones vectoriales

**Modelo elegido**: `sentence-transformers/all-MiniLM-L6-v2`

**Justificación**:
- - Balance perfecto entre calidad y velocidad
- - Dimensión: 384 (eficiente en memoria)
- - Multilenguaje (español e inglés)
- - Pre-entrenado en tareas de similitud semántica
- - No requiere GPU (puede correr en CPU)
- - Gratuito y open-source

**Alternativas consideradas**:
- `text-embedding-ada-002` (OpenAI): Mejor calidad pero costoso
- `bge-large`: Mayor precisión pero más lento
- `all-mpnet-base-v2`: Más preciso pero más pesado

### 3. Vector Store (`vector_store.py`)

**Responsabilidad**: Almacenamiento e indexación vectorial

**Tecnología elegida**: FAISS (Facebook AI Similarity Search)

**Justificación**:
- - Extremadamente rápido para búsqueda de similitud
- - Soporta millones de vectores en memoria
- - No requiere servidor externo
- - Gratuito y open-source
- - Integración simple con NumPy

**Configuración**:
- Índice: `IndexFlatIP` (Inner Product para cosine similarity)
- Normalización L2 de vectores para similitud coseno
- Serialización con pickle para persistencia

**Alternativas consideradas**:
- Pinecone: Más escalable pero requiere servicio cloud
- Weaviate: Más features pero mayor complejidad
- Chroma: Similar a FAISS pero menos maduro

### 4. RAG Engine (`rag_engine.py`)

**Responsabilidad**: Orquestación del pipeline RAG

**LLM elegido**: OpenAI GPT-3.5-turbo (con fallback)

**Justificación**:
- - Excelente calidad de respuestas
- - Sigue instrucciones precisamente
- - Bajo costo por token
- - API estable y bien documentada

**Control de Alucinaciones**:
1. **Prompt Engineering**: Instrucciones explícitas de usar solo el contexto
2. **Relevance Threshold**: Solo responder si similarity > 0.3
3. **Confidence Scoring**: Basado en scores de similitud
4. **Fallback Messages**: Mensajes claros cuando no hay información

**Alternativas consideradas**:
- GPT-4: Mayor precisión pero 10x más costoso
- Llama 2: Gratuito pero requiere GPU y es más lento
- Claude: Excelente pero menos accesible

### 5. API REST (`api.py`)

**Responsabilidad**: Interfaz HTTP para el sistema

**Framework elegido**: FastAPI

**Justificación**:
- - Alto rendimiento (comparable a Node.js)
- - Type hints nativos con Pydantic
- - Documentación automática (Swagger/ReDoc)
- - Async/await nativo
- - Fácil deployment

**Endpoints**:
- `POST /ask`: Enviar pregunta y recibir respuesta
- `GET /health`: Verificar estado del sistema
- `GET /stats`: Estadísticas del vector store

---

## Flujo de Datos

### Pipeline de Ingesta

```
1. Usuario proporciona documento
   ↓
2. DocumentProcessor extrae texto
   ↓
3. Limpieza y normalización
   ↓
4. División en chunks (500 chars, overlap 50)
   ↓
5. EmbeddingGenerator crea vectores 384-dim
   ↓
6. VectorStore indexa con FAISS
   ↓
7. Persistencia en disco
```

### Pipeline de Consulta

```
1. Usuario hace pregunta
   ↓
2. API recibe request en /ask
   ↓
3. RAGEngine genera embedding de la pregunta
   ↓
4. VectorStore busca top-K chunks más similares
   ↓
5. RAGEngine evalúa calidad del contexto
   ↓
6. Si calidad suficiente:
   ├─► Construye prompt con contexto
   ├─► Llama a LLM (GPT-3.5)
   └─► Retorna respuesta + fuentes

   Si calidad insuficiente:
   └─► Retorna mensaje de información insuficiente
```

---

## Decisiones de Diseño

### 1. Chunking Strategy

**Decisión**: Chunking basado en sentencias con overlap

**Razones**:
- Mantiene coherencia semántica
- El overlap evita pérdida de contexto en límites
- Tamaño de 500 chars balancea contexto vs. precisión

### 2. Embedding Normalization

**Decisión**: Normalización L2 de todos los vectores

**Razones**:
- Permite usar cosine similarity eficientemente
- FAISS Inner Product equivale a cosine con vectores normalizados
- Mejora la comparabilidad de scores

### 3. Stateless API

**Decisión**: API sin estado de sesión

**Razones**:
- Más fácil de escalar horizontalmente
- Menor complejidad
- Cada request es independiente

### 4. Lazy Loading

**Decisión**: Vector store se carga en startup

**Razones**:
- Evita latencia en primera consulta
- Mejor experiencia de usuario
- Fácil health check

---

## Limitaciones

### Limitaciones Actuales

1. **Escalabilidad de Vector Store**
   - FAISS en memoria limita a ~10M vectores
   - No distribuido

2. **Sin Memoria Conversacional**
   - No mantiene historial de conversación
   - Cada pregunta es independiente

3. **Modelo de Embeddings Fijo**
   - Cambiar modelo requiere re-indexar todo

4. **Sin Caché de Respuestas**
   - Preguntas idénticas re-procesan todo

5. **Control de Alucinaciones Básico**
   - Solo threshold simple de relevancia
   - No verifica factualidad

6. **Monolítico**
   - Todos los componentes en un proceso
   - No microservicios

### Mitigaciones Posibles

1. **Vector Store**: Migrar a Pinecone/Weaviate para escala
2. **Memoria**: Implementar sesiones con Redis
3. **Modelo**: Arquitectura plugin para múltiples modelos
4. **Caché**: Redis para preguntas frecuentes
5. **Alucinaciones**: Fact-checking con múltiples fuentes
6. **Arquitectura**: Separar en microservicios

---

## Escalabilidad en Producción

### Arquitectura Escalable Propuesta

```
┌─────────────────────────────────────────────────────────────┐
│                      LOAD BALANCER                          │
│                     (NGINX/AWS ALB)                         │
└────────────┬────────────────────────────────┬───────────────┘
             │                                │
             ▼                                ▼
    ┌────────────────┐              ┌────────────────┐
    │  API Instance  │              │  API Instance  │
    │   (FastAPI)    │              │   (FastAPI)    │
    └────────┬───────┘              └────────┬───────┘
             │                                │
             └────────────┬───────────────────┘
                          ▼
                 ┌────────────────┐
                 │  Redis Cache   │
                 │  (Responses)   │
                 └────────────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │   Pinecone     │
                 │ (Vector Store) │
                 └────────────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │   PostgreSQL   │
                 │  (Metadata)    │
                 └────────────────┘
```

### Estrategias de Escalado

#### Horizontal Scaling

1. **Múltiples instancias de API**
   - Stateless permite N réplicas
   - Load balancer distribuye tráfico
   - Auto-scaling basado en CPU/requests

2. **Vector Store Distribuido**
   - Migrar a Pinecone/Weaviate
   - Sharding por namespace
   - Replicación para alta disponibilidad

#### Vertical Scaling

1. **Optimización de Embeddings**
   - Batch processing de queries
   - GPU para modelos más grandes
   - Cuantización de vectores

2. **Caché Inteligente**
   - Redis para respuestas frecuentes
   - TTL basado en cambios de documentos
   - Cache warming para queries comunes

### Infrastructure as Code

```yaml
# docker-compose.yml example
version: '3.8'
services:
  api:
    build: .
    replicas: 3
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://cache:6379
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    depends_on:
      - cache
      - postgres

  cache:
    image: redis:7-alpine

  postgres:
    image: postgres:15

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    depends_on:
      - api
```

### Cloud Deployment

**AWS**:
- ECS/EKS para containers
- Application Load Balancer
- ElastiCache Redis
- RDS PostgreSQL
- S3 para documentos

**GCP**:
- Cloud Run / GKE
- Cloud Load Balancing
- Memorystore Redis
- Cloud SQL
- Cloud Storage

---

## Memoria Conversacional

### Implementación Propuesta

```python
# Estructura de sesión
{
  "session_id": "uuid",
  "messages": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "assistant", "content": "...", "timestamp": "..."}
  ],
  "context_history": [...],
  "metadata": {
    "created_at": "...",
    "last_active": "..."
  }
}
```

### Estrategias

1. **Short-term Memory (Redis)**
   - Últimas 10 interacciones
   - TTL de 30 minutos
   - Rápido acceso

2. **Long-term Memory (PostgreSQL)**
   - Historial completo
   - Análisis de patrones
   - Mejora continua

3. **Context Window Management**
   - Sliding window de mensajes
   - Summarization de conversaciones largas
   - Re-ranking de contexto relevante

### Flujo con Memoria

```
1. Usuario hace pregunta
   ↓
2. Recuperar historial de sesión
   ↓
3. Generar embedding considerando contexto previo
   ↓
4. Buscar en vector store
   ↓
5. Re-rank resultados con historial
   ↓
6. Construir prompt con:
   - Historial de conversación
   - Contexto recuperado
   - Pregunta actual
   ↓
7. Generar respuesta
   ↓
8. Guardar en historial
```

---

## Integración con Canales

### WhatsApp

```python
# Usando Twilio/WhatsApp Business API
from twilio.rest import Client

def handle_whatsapp_message(from_number, message):
    # 1. Obtener/crear sesión
    session = get_or_create_session(from_number)

    # 2. Llamar RAG engine
    response = rag_engine.answer_question(message)

    # 3. Enviar respuesta
    client.messages.create(
        from_='whatsapp:+1234567890',
        to=from_number,
        body=response['answer']
    )
```

### Web Chat Widget

```javascript
// Frontend integration
async function askQuestion(question) {
  const response = await fetch('https://api.example.com/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question})
  });

  const data = await response.json();
  displayAnswer(data.answer);
}
```

### Slack Bot

```python
# Usando Bolt for Python
from slack_bolt import App

app = App(token=os.environ["SLACK_BOT_TOKEN"])

@app.message(".*")
def handle_message(message, say):
    response = rag_engine.answer_question(message['text'])
    say(response['answer'])
```

### Arquitectura Multi-Canal

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  WhatsApp   │  │  Web Chat   │  │    Slack    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Channel Router │
              │  (Normalizer)   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   RAG API       │
              └─────────────────┘
```

---

## Monitoreo y Observabilidad

### Métricas Clave

#### Performance Metrics

1. **Latencia**
   - P50, P95, P99 de respuestas
   - Breakdown por componente (embedding, search, LLM)
   - Target: < 2s end-to-end

2. **Throughput**
   - Requests por segundo
   - Concurrent users
   - Target: 100 req/s

3. **Error Rate**
   - 4xx vs 5xx errors
   - Tipos de errores
   - Target: < 1% error rate

#### Business Metrics

1. **Engagement**
   - Preguntas por sesión
   - Tiempo de sesión
   - Retención de usuarios

2. **Quality**
   - Confidence scores promedio
   - Porcentaje de "información insuficiente"
   - Feedback explícito de usuarios

3. **Costos**
   - Tokens consumidos (OpenAI)
   - Costo por query
   - ROI vs alternativas

### Stack de Monitoreo Propuesto

```yaml
Métricas: Prometheus
Visualización: Grafana
Logs: ELK Stack (Elasticsearch, Logstash, Kibana)
Tracing: Jaeger / OpenTelemetry
Alertas: PagerDuty / Slack
```

### Logging Structure

```python
import structlog

logger = structlog.get_logger()

logger.info(
    "question_answered",
    session_id=session_id,
    question_length=len(question),
    num_chunks_retrieved=len(chunks),
    confidence=confidence,
    latency_ms=latency,
    llm_tokens=tokens_used,
    cache_hit=cache_hit
)
```

### Dashboard Example

```
┌─────────────────────────────────────────────────────────────┐
│                   RAG Chatbot Dashboard                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Requests/min: 45  │  Avg Latency: 1.2s  │  Error Rate: 0.3% │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Response Time Distribution (P95)            │   │
│  │  [████████████████░░░░░░░░░░░░] 1.8s               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Confidence Score Distribution             │   │
│  │  High (>0.7):   ████████ 65%                        │   │
│  │  Medium (0.3-0.7): ████ 25%                         │   │
│  │  Low (<0.3):    ██ 10%                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Top Errors:                                                │
│  - Vector store timeout: 5                                  │
│  - OpenAI rate limit: 2                                     │
│  - Invalid request: 1                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Alertas Críticas

1. **Error rate > 5%** → PagerDuty inmediato
2. **Latency P95 > 5s** → Slack warning
3. **OpenAI API down** → Failover a modelo local
4. **Vector store unavailable** → Critical alert
5. **Disk space < 10%** → Warning

---

## Seguridad y Compliance

### Consideraciones de Seguridad

1. **API Security**
   - Rate limiting (100 req/min por IP)
   - API keys para autenticación
   - HTTPS obligatorio
   - CORS configurado apropiadamente

2. **Data Privacy**
   - No guardar preguntas sensibles
   - Encriptación en tránsito y reposo
   - GDPR compliance (derecho al olvido)
   - Anonimización de logs

3. **Input Validation**
   - Sanitización de inputs
   - Límite de tamaño de queries
   - Prevención de injection attacks

---

## Roadmap Futuro

### Fase 1 (1-3 meses)
- - MVP funcional
- - API REST básica
- - Integración OpenAI
- - Tests automatizados
- - CI/CD pipeline

### Fase 2 (3-6 meses)
- - Memoria conversacional
- - Múltiples modelos de embeddings
- - Caché con Redis
- - Integración WhatsApp
- - Dashboard de métricas

### Fase 3 (6-12 meses)
- - Multi-tenant support
- - Fine-tuning de modelos
- - A/B testing framework
- - Feedback loop automático
- - Vector store distribuido

---

## Referencias

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

---

**Documento versión**: 1.0
**Última actualización**: 2024
**Autor**: Prueba Técnica - Ingeniero IA/ML
