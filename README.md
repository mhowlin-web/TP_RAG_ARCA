# Asistente RAG para consultas sobre trámites de Monotributo en ARCA

## Descripción del proyecto

En este proyecto implemento un sistema de **Retrieval-Augmented Generation (RAG)** para responder preguntas sobre trámites relacionados con el **Monotributo** utilizando información proveniente de fuentes oficiales de **ARCA (Agencia de Recaudación y Control Aduanero)**.

El objetivo es construir un asistente capaz de responder consultas en lenguaje natural utilizando como fuente de conocimiento un corpus documental específico. De esta manera, el modelo de lenguaje no depende exclusivamente del conocimiento aprendido durante su entrenamiento, sino que recibe como contexto información recuperada desde una base vectorial.

El sistema permite realizar consultas como:

* ¿Cómo puedo obtener la clave fiscal?
* ¿Cómo puedo realizar la recategorización del Monotributo?
* ¿Cómo puedo dar de baja el Monotributo?
* ¿Qué información tiene ARCA sobre la facturación del Monotributo?

La arquitectura implementada utiliza:

* **ARCA** como fuente de información.
* **Sentence Transformers** para generar embeddings.
* **Pinecone** como base de datos vectorial.
* **Hugging Face Inference API** para acceder al modelo de lenguaje.
* **Google Colab** como entorno de ejecución.

---

# 1. Fundamentos teóricos

## ¿Qué es RAG?

RAG significa **Retrieval-Augmented Generation**, o generación aumentada mediante recuperación.

Es una arquitectura que combina dos procesos principales:

1. **Retrieval (recuperación):** buscar información relevante dentro de una colección de documentos.
2. **Generation (generación):** utilizar esa información recuperada como contexto para que un modelo de lenguaje genere una respuesta.

Un modelo de lenguaje tradicional intenta responder una pregunta utilizando principalmente el conocimiento adquirido durante su entrenamiento.

En cambio, en un sistema RAG la pregunta primero se utiliza para recuperar información relevante desde una fuente externa. Esa información se incorpora posteriormente al prompt enviado al modelo.

El flujo conceptual es:

```text
Pregunta del usuario
        ↓
Embedding de la pregunta
        ↓
Búsqueda vectorial
        ↓
Documentos relevantes
        ↓
Contexto
        ↓
Modelo de lenguaje
        ↓
Respuesta
```

Una ventaja importante de este enfoque es que permite utilizar información específica y actualizable sin necesidad de volver a entrenar el modelo de lenguaje.

En este proyecto utilizo esta característica para restringir las respuestas a información proveniente del corpus documental de ARCA.

---

## Embeddings

Para realizar la recuperación semántica necesito representar tanto los documentos como las preguntas mediante vectores numéricos.

Estos vectores reciben el nombre de **embeddings**.

En el proyecto utilizo:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Este modelo transforma un fragmento de texto en un vector.

La pregunta del usuario se transforma utilizando el mismo modelo utilizado para generar los embeddings de los documentos.

Esto permite comparar la representación vectorial de la pregunta con las representaciones vectoriales almacenadas en Pinecone.

---

## Base de datos vectorial

Los embeddings de los documentos se almacenan en **Pinecone**, una base de datos especializada en búsqueda vectorial.

Cuando el usuario realiza una consulta, genero el embedding correspondiente a la pregunta y lo envío a Pinecone.

Pinecone busca los vectores más similares y devuelve los fragmentos correspondientes.

En el proyecto utilizo una estrategia **top-k**, recuperando los cinco fragmentos más similares a cada consulta.

---

## Chunking

Los documentos originales son páginas web relativamente extensas.

Por este motivo no resulta conveniente enviar un documento completo al modelo de lenguaje cada vez que se realiza una consulta.

Primero divido los documentos en fragmentos o **chunks**.

En este proyecto el corpus procesado genera:

```text
43 chunks
```

Cada chunk es posteriormente transformado en un embedding y almacenado en Pinecone.

El chunking permite realizar una recuperación más precisa, ya que Pinecone puede recuperar solamente las partes del documento relacionadas con la pregunta.

---

## Generación aumentada

Una vez recuperados los chunks relevantes, construyo un contexto que contiene esos fragmentos.

El prompt enviado al modelo de lenguaje contiene:

```text
CONTEXTO:
[fragmentos recuperados]

PREGUNTA:
[pregunta del usuario]
```

Además, establezco explícitamente que el modelo debe utilizar únicamente la información proporcionada en el contexto y que no debe inventar información.

Esto busca reducir el riesgo de que el modelo responda utilizando conocimiento externo al corpus.

---

# 2. Corpus documental

El corpus está construido a partir de páginas oficiales de ARCA relacionadas con el Monotributo.

Entre las fuentes utilizadas se encuentran:

* Inicio y procedimiento general del Monotributo.
* Obtención de Clave Fiscal.
* Constancias y credenciales.
* Facturación.
* Recategorización.
* Baja del Monotributo.
* Desarrollo de la actividad.
* Tutoriales sobre Monotributo.

La información se descarga automáticamente desde las fuentes oficiales y se almacena inicialmente en formato JSON.

El corpus conserva metadatos como:

* identificador del documento;
* título;
* organismo;
* URL de origen;
* fecha de descarga;
* contenido textual.

Esto permite mantener la trazabilidad entre la información recuperada y su fuente original.

---

# 3. Arquitectura del sistema

La implementación está organizada en diferentes notebooks que representan las distintas etapas del pipeline.

## Notebook 01 — Descarga del corpus

```text
01_descarga_corpus_arca.ipynb
```

En este notebook descargo las páginas oficiales de ARCA, extraigo su contenido textual y construyo el corpus documental.

El resultado es un archivo JSON que contiene los documentos y sus metadatos.

---

## Notebook 02 — Corrección y limpieza

En esta etapa realizo el procesamiento necesario sobre el corpus para corregir problemas de codificación de caracteres y obtener un texto adecuado para las etapas posteriores.

Esto es particularmente importante porque los documentos contienen caracteres propios del idioma español, como tildes y caracteres especiales.

---

## Notebook 03 — Chunking

En esta etapa divido los documentos en fragmentos más pequeños.

El corpus procesado queda compuesto por:

```text
43 chunks
```

Estos fragmentos constituyen las unidades que posteriormente serán vectorizadas.

---

## Notebook 04 — Embeddings

En esta etapa genero los embeddings utilizando:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Cada chunk se transforma en un vector numérico.

El resultado es un conjunto de:

```text
43 embeddings
```

---

## Notebook 05 — Pinecone

En esta etapa creo y cargo los embeddings en un índice específico de Pinecone:

```text
arca-monotributo
```

No utilizo el índice `tutorial` utilizado durante las pruebas iniciales del aprendizaje.

Pinecone pasa a funcionar como la base vectorial del sistema.

---

## Notebook 06 — Retrieval

En esta etapa implemento la recuperación semántica.

La pregunta del usuario se transforma en un embedding y se consulta el índice de Pinecone.

El sistema recupera los cinco chunks más similares:

```text
TOP_K = 5
```

---

## Notebook 07 — RAG completo

En este notebook implemento una primera versión completa del pipeline RAG utilizando un modelo de lenguaje ejecutado localmente.

Esta versión permitió verificar el funcionamiento de la arquitectura completa antes de utilizar un modelo mediante API.

---

## Notebook 08 — RAG final

```text
08_RAG_HuggingFace_API.ipynb
```

Este notebook constituye la versión final del RAG presentado en el TP.

En esta versión mantengo Pinecone para la recuperación de información y utilizo la **Hugging Face Inference API** para acceder al modelo de lenguaje.

Por lo tanto, el modelo generativo no se ejecuta localmente en Google Colab.

---

# 4. Flujo completo de ejecución

El pipeline final funciona de la siguiente manera:

```text
                   CORPUS ARCA
                       │
                       ▼
                Limpieza de texto
                       │
                       ▼
                    Chunking
                       │
                       ▼
                   Embeddings
                       │
                       ▼
                  ┌───────────┐
                  │ Pinecone  │
                  └───────────┘
                       ▲
                       │
Pregunta ──► Embedding ┘
                       │
                       ▼
                  Top-K chunks
                       │
                       ▼
                    Contexto
                       │
                       ▼
              Hugging Face API
                       │
                       ▼
                   Respuesta
```

La separación entre recuperación y generación permite distinguir claramente las dos partes principales del sistema.

---

# 5. Instalación

El proyecto está pensado para ejecutarse en **Google Colab**.

Las principales librerías utilizadas son:

```bash
pip install pinecone sentence-transformers huggingface_hub
```

Para la construcción inicial del corpus también se utilizan:

```bash
pip install requests beautifulsoup4 lxml
```

---

# 6. Configuración de credenciales

El proyecto utiliza dos servicios externos:

* Pinecone.
* Hugging Face.

Las claves de acceso no están escritas directamente en los notebooks.

En Google Colab se deben configurar como **Secrets**.

Los nombres utilizados por el proyecto son:

```text
PINECONE_API_KEY
HUGGINGFACE_TOKEN
```

El notebook recupera estas credenciales mediante:

```python
from google.colab import userdata

PINECONE_API_KEY = userdata.get("PINECONE_API_KEY")
HF_TOKEN = userdata.get("HUGGINGFACE_TOKEN")
```

De esta manera las claves no forman parte del código que se publica en GitHub.

---

# 7. Ejecución

Para reproducir el proyecto se deben ejecutar los notebooks en orden.

### Paso 1

Ejecutar:

```text
01_descarga_corpus_arca.ipynb
```

Esto construye el corpus documental.

### Paso 2

Ejecutar el notebook de limpieza del corpus.

### Paso 3

Ejecutar el notebook de chunking.

### Paso 4

Ejecutar el notebook de generación de embeddings.

### Paso 5

Ejecutar el notebook de carga de embeddings en Pinecone.

En esta etapa se crea o utiliza el índice:

```text
arca-monotributo
```

### Paso 6

Ejecutar el notebook de retrieval para verificar la recuperación.

### Paso 7

Ejecutar:

```text
07_RAG_completo.ipynb
```

para comprobar el funcionamiento de la arquitectura RAG.

### Paso 8

Ejecutar:

```text
08_RAG_HuggingFace_API.ipynb
```

Este es el notebook final del proyecto.

---

# 8. Ejemplos de uso

## Ejemplo 1 — Clave Fiscal

Pregunta:

```text
¿Cómo puedo obtener la clave fiscal?
```

El sistema recupera fragmentos relacionados con la obtención de la Clave Fiscal y los utiliza como contexto para el modelo.

La respuesta generada puede indicar que la clave fiscal puede obtenerse mediante la aplicación móvil ARCA utilizando la opción correspondiente y mencionar el procedimiento alternativo indicado en los documentos recuperados.

---

## Ejemplo 2 — Recategorización

Pregunta:

```text
¿Cómo puedo realizar la recategorización del Monotributo?
```

El sistema realiza la búsqueda semántica en Pinecone y recupera los fragmentos relacionados con la recategorización.

---

## Ejemplo 3 — Baja

Pregunta:

```text
¿Cómo puedo dar de baja el Monotributo?
```

La pregunta se transforma en un embedding y se utiliza para recuperar los fragmentos más similares del corpus.

---

# 9. Evaluación del retrieval

Durante las pruebas observo que las preguntas relacionadas con el corpus producen scores de similitud superiores a los obtenidos por una pregunta completamente ajena al corpus.

Por ejemplo:

| Pregunta                                       | Mejor score |
| ---------------------------------------------- | ----------: |
| ¿Cómo puedo obtener la clave fiscal?           |      0.6147 |
| ¿Cómo puedo realizar la recategorización?      |      0.7032 |
| ¿Cómo puedo dar de baja el Monotributo?        |      0.7072 |
| ¿Qué información tiene ARCA sobre facturación? |      0.5664 |
| ¿Quién fue el presidente de Argentina en 1990? |      0.3798 |

Esto muestra que la búsqueda vectorial encuentra mayor similitud para consultas relacionadas con el dominio documental.

Sin embargo, la búsqueda vectorial siempre devuelve los vectores más similares, incluso cuando la pregunta no pertenece al corpus. Por este motivo, un sistema RAG robusto debería incorporar mecanismos adicionales para determinar cuándo el contexto recuperado es suficientemente relevante.

---

# 10. Análisis crítico

## Limitaciones

### Tamaño del corpus

El corpus utilizado es relativamente pequeño y está compuesto por páginas seleccionadas de ARCA relacionadas con el Monotributo.

Aunque cumple con el objetivo del TP y permite demostrar el funcionamiento de RAG, no representa toda la documentación disponible de ARCA.

Una versión de producción debería incorporar una colección documental mucho más amplia y actualizada.

### Actualización de la información

La información se obtiene de páginas web oficiales en un momento determinado.

Los trámites y procedimientos administrativos pueden cambiar.

Por lo tanto, el sistema debería contar con un mecanismo periódico de actualización del corpus.

### Calidad de la extracción

El contenido se obtiene a partir de páginas HTML.

La extracción automática puede incorporar elementos de navegación o perder parte de la estructura original de las páginas.

También pueden aparecer problemas de codificación si las fuentes no son procesadas correctamente.

### Recuperación

Actualmente utilizo una estrategia sencilla de recuperación basada en similitud vectorial y `top-k`.

Esto significa que Pinecone devuelve los cinco fragmentos más similares independientemente de que todos sean realmente necesarios para responder la pregunta.

Una mejora posible sería incorporar un **umbral de similitud** para descartar resultados poco relevantes.

### Generación

El modelo de lenguaje puede generar una respuesta incorrecta aunque el contexto recuperado sea correcto.

Las instrucciones del prompt reducen este problema al exigir que el modelo utilice solamente el contexto proporcionado, pero no eliminan completamente el riesgo.

### Trazabilidad de las respuestas

El sistema conserva la información de origen de los documentos, pero una versión más avanzada podría mostrar explícitamente al usuario las fuentes utilizadas para generar cada respuesta.

---

# 11. Mejoras futuras

Entre las mejoras que considero más importantes se encuentran:

1. **Ampliar el corpus**, incorporando más documentación oficial de ARCA.
2. **Actualizar automáticamente los documentos** para evitar información desactualizada.
3. **Optimizar el chunking**, evaluando distintos tamaños y solapamientos.
4. **Incorporar un umbral de similitud** para evitar respuestas cuando no existe suficiente evidencia.
5. **Evaluar diferentes modelos de embeddings**.
6. **Comparar diferentes modelos de lenguaje** mediante Hugging Face.
7. **Incorporar las fuentes recuperadas en la respuesta final** para mejorar la trazabilidad.
8. **Construir un conjunto de evaluación más amplio**, con preguntas dentro y fuera del dominio.
9. **Comparar sistemáticamente respuestas con RAG y sin RAG**.

---

# 12. Conclusiones

En este proyecto implementé un sistema RAG funcional para responder consultas sobre trámites de Monotributo utilizando documentación oficial de ARCA.

El sistema integra las principales etapas de una arquitectura RAG:

```text
Documentos
    ↓
Chunking
    ↓
Embeddings
    ↓
Base vectorial
    ↓
Retrieval Top-K
    ↓
Contexto
    ↓
LLM
    ↓
Respuesta
```

La base vectorial utilizada es Pinecone y el modelo de lenguaje se consume mediante la API de Hugging Face.

La implementación permite separar claramente la recuperación de información de la generación de lenguaje. Esto hace posible que el modelo responda utilizando información específica del dominio seleccionado en lugar de depender exclusivamente de su conocimiento general.

El proyecto constituye una implementación práctica de la arquitectura **Retrieval-Augmented Generation (RAG)** y permite analizar tanto sus ventajas como sus limitaciones en un caso de uso concreto: la consulta de trámites administrativos de ARCA.
