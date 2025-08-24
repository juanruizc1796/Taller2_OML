# Taller 2

Este proyecto implementa un flujo de trabajo de MLOps para el entrenamiento y despliegue de modelos de Machine Learning. La arquitectura separa el entorno de experimentación del de inferencia, permitiendo que la API consuma nuevos modelos de forma dinámica, sin necesidad de reconstruir o reiniciar los servicios.

## Arquitectura del Proyecto

Este proyecto utiliza una arquitectura de microservicios orquestada por Docker Compose, compuesta por tres elementos clave:
- **Contenedro de Entranamiento:** Contiene las librerias necesarias para el entrenamiento de modelos, Donde se generaran los modelos.

- **Contenedor de Inferencia:** Contruido con FastAPI, con la finalidad de exponer endpoints para cargar modelos y suministrar predicciones.

- **Volumen:** Funciona como un sistema de archivos compartidos que persiste los datos. Este se monta en ambos contenedores, con la finalidad que el contenedor de entranamiento guarde los modelos para que el 

## Estructura de archivos

El proyecto está organizado de la siguiente manera:

````txt
.
├── docker-compose.yml
├── fastapi_app/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
└── jupyter_lab/
    ├── Dockerfile
    └── requirements.txt
````

## Descripción Detallada de Archivos

**docker-compose.yml:** Define y orquesta los servicios que componen la aplicación, permitiéndoles comunicarse entre sí. Al ejecutar `docker-compose up`, este archivo le dice a Docker cómo construir cada imagen, qué puertos exponer y, crucialmente, cómo compartir datos entre los contenedores a través del volumen model_storage.

**Directorio `fastapi_app/`**: Este directorio contiene todo lo necesario para el servicio de inferencia, es decir, el servidor que se encarga de suministrar predicciones.

- **Dockerfile:** Es la receta para construir la imagen de Docker para la API de producción. Utiliza una imagen base ligera de Python (3.11-slim) y solo instala las dependencias estrictamente necesarias para el servidor.

- **requirements.txt:** Lista las librerías de Python requeridas para el servidor, como fastapi, uvicorn y joblib, que se usan para cargar el modelo.

- **main.py:** Contiene la lógica principal de la API con FastAPI. Sus endpoints (/models, models/{model_name}/activate, /predict) permiten listar, seleccionar y usar dinámicamente los modelos almacenados.

**Directorio `jupyter_lab/`**:

Este directorio se dedica al entorno de desarrollo y experimentación.

- **Dockerfile:** La receta para construir el entorno de laboratorio. Este Dockerfile instala librerías de análisis y entrenamiento de datos más completas.

- **requirements.txt:** Lista las dependencias de Python necesarias para el desarrollo, como jupyterlab, pandas, scikit-learn y palmerpenguins.

## Creación de la estructura de archivos.

En la terminal realizamos la creación de los archivos:
````bash
mkdir taller_mlops_final
cd taller_mlops_final

mkdir jupyter_lab
touch jupyter_lab/Dockerfile
touch jupyter_lab/requirements.txt

mkdir fastapi_app
touch fastapi_app/Dockerfile
touch fastapi_app/requirements.txt
touch fastapi_app/main.py

touch docker-compose.yml
````

## Estructura de Archivos

### 1. Preparación de los Contenedores

**`jupyter_lab/requirements.txt`**: Dependencias necesarias para el análisis y entrenamiento de modelos:

```text
jupyterlab
pandas
scikit-learn
requests
palmerpenguins
```

**`jupyter_lab/Dockerfile`**: Receta para construir el contenedor de JupyterLab:

```dockerfile
# Utiliza una imagen ligera de Python como base.
FROM python:3.11-slim
# Establece el directorio de trabajo dentro del contenedor.
WORKDIR /app
# Instala el gestor de paquetes uv para instalaciones rápidas.
RUN pip install uv
# Copia el archivo de dependencias del host al contenedor.
COPY requirements.txt .
# Instala las dependencias del archivo de requisitos.
RUN uv pip install --system --no-cache-dir -r requirements.txt
# Expone el puerto 8888, que es el de JupyterLab.
EXPOSE 8888
# Comando para iniciar JupyterLab, haciéndolo accesible desde cualquier IP.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
```

**`fastapi_app/requirements.txt`**: Dependencias mínimas para la API de inferencia:

```text
fastapi
uvicorn[standard]
scikit-learn
joblib
```

**`fastapi_app/Dockerfile`**: Receta para construir el contenedor de la API:

```dockerfile
# Utiliza una imagen ligera de Python como base.
FROM python:3.11-slim
# Establece el directorio de trabajo dentro del contenedor.
WORKDIR /app
# Instala el gestor de paquetes uv para instalaciones rápidas.
RUN pip install uv
# Copia el archivo de dependencias.
COPY requirements.txt .
# Instala las dependencias del archivo de requisitos.
RUN uv pip install --system --no-cache-dir -r requirements.txt
# Copia el código de la API al contenedor.
COPY main.py .
# Expone el puerto 8000, que es el de la API.
EXPOSE 8000
# Comando para iniciar el servidor uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```


### 2. Unificación con Docker Compose

**`docker-compose.yml`**: Archivo orquestador que conecta los servicios y define el volumen compartido:

```yaml
# Especifica la versión de Docker Compose.
version: '3.8'
services:
  # Define el servicio de Jupyter Lab.
  jupyter_lab:
    # Construye la imagen usando el Dockerfile en ./jupyter_lab.
    build: ./jupyter_lab
    # Mapea el puerto del host al puerto del contenedor.
    ports:
      - "8888:8888"
    # Monta el volumen 'model_storage' en la ruta /app/models.
    volumes:
      - model_storage:/app/models

  # Define el servicio de la API de FastAPI.
  fastapi_app:
    # Construye la imagen usando el Dockerfile en ./fastapi_app.
    build: ./fastapi_app
    # Mapea el puerto del host al puerto del contenedor.
    ports:
      - "8000:8000"
    # Monta el mismo volumen 'model_storage' en la misma ruta.
    volumes:
      - model_storage:/app/models

# Declara el volumen para que sea persistente y compartido.
volumes:
  model_storage:
```


### 3. Lógica del Flujo de MLOps

**`fastapi_app/main.py`**:Código de la API que carga modelos dinámicamente y realiza predicciones:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import glob

# Inicialización de la aplicación FastAPI.
app = FastAPI(title="API con Selección Dinámica de Modelos")

# Constantes y variables globales.
MODEL_DIR = "/app/models"  # Ruta al volumen compartido donde se guardan los modelos.
active_model = None        # Variable global para mantener el modelo cargado en memoria.
active_model_name = "Ninguno"

def get_available_models():
    # Escanea la carpeta de modelos y devuelve una lista de nombres de archivo.
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.joblib"))
    return [os.path.basename(f) for f in model_files]

@app.get("/models", tags=["Gestión de Modelos"])
async def list_available_models():
    # Devuelve una lista de todos los modelos disponibles en el volumen.
    models = get_available_models()
    return {
        "active_model": active_model_name,
        "available_models": models
    }

@app.post("/models/{model_name}/activate", tags=["Gestión de Modelos"])
async def activate_model(model_name: str):
    # Carga un modelo específico por su nombre de archivo para que sea el 'activo'.
    global active_model, active_model_name

    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")

    try:
        # Carga el modelo desde el volumen compartido.
        active_model = joblib.load(model_path)
        active_model_name = model_name
        print(f"Modelo activado: {active_model_name}")
        return {"message": f"Modelo '{model_name}' activado exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo: {e}" )

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict", tags=["Inferencia"])
async def predict(request: PredictionRequest):
    # Realiza una predicción usando el modelo actualmente activo.
    if active_model is None:
        raise HTTPException(status_code=503, detail="Ningún modelo está activo. Por favor, active uno primero.")

    # Usa el modelo que está cargado en memoria para hacer la predicción.
    prediction = active_model.predict([request.features])
    return {
        "prediction": prediction.tolist(),
        "model_used": active_model_name
    }
```

##  Guía de Uso

### 1. Construir y Lanzar el Entorno
En la carpeta raíz del proyecto, ejecutar:

```bash
docker-compose up --build
```

### 2. Entrenar Modelos
- Accede a Jupyter en: [http://localhost:8888](http://localhost:8888)  
- Abre el notebook `model.ipynb` y entrena un modelo.
**`model.ipynb`**:
````python
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os
import glob
import itertools

#########################################
# 1. PREPARACIÓN DE DATOS (Sin cambios) #
#########################################

df = load_penguins()
df.dropna(inplace=True)
X = df[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
y = LabelEncoder().fit_transform(df["species"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#########################################
# 2. LIMPIAR MODELOS ANTERIORES (Nuevo) #
#########################################

model_dir = '/app/models'
os.makedirs(model_dir, exist_ok=True) 

# Borra todos los archivos .joblib que ya existían
print("Limpiando modelos antiguos")
for old_model in glob.glob(os.path.join(model_dir, "*.joblib")):
    os.remove(old_model)
print("Carpeta de modelos limpia.")

##########################
# 3. DEFINIR EXPERIMENTO #  
##########################

param_grid = {
    'n_estimators': [50, 75, 100],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 2, 4]
}

# Crear todas las combinaciones posibles de hiperparámetros
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

results = []
print(f"\nIniciando entrenamiento de {len(param_combinations)} modelos")

##########################################
# 4. BUCLE DE ENTRENAMIENTO Y EVALUACIÓN # 
##########################################

for i, params in enumerate(param_combinations):
    print(f"\n--- Entrenando Modelo {i+1}/{len(param_combinations)} ---")
    print(f"Parámetros: {params}")

    # Entrenar el modelo con los parámetros actuales
    rf = RandomForestClassifier(random_state=42, **params)
    rf.fit(X_train, y_train)

    # Evaluar el modelo en el conjunto de prueba
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Exactitud (Accuracy): {accuracy:.4f}")

    # Guardar el modelo con un nombre descriptivo
    # Incluye los parámetros y la métrica en el nombre del archivo
    model_name = f"rf_n{params['n_estimators']}_md{params['max_depth'] or 'None'}_msl{params['min_samples_leaf']}_acc{accuracy:.2f}.joblib"
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(rf, model_path)
    print(f"Modelo guardado en: {model_path}")

    # Guardar resultados para la tabla final
    results.append({**params, 'accuracy': accuracy, 'filename': model_name})

################################
# 5. MOSTRAR TABLA COMPARATIVA #
################################

print("\n--- Resultados del Experimento ---")
results_df = pd.DataFrame(results)
print(results_df.sort_values(by='accuracy', ascending=False).reset_index(drop=True))
````  

### 3. Interactuar con la API 
- **Listar modelos**: `GET /models` ejecutar en la terminal
````bash
curl http://localhost:8000/models
````
- **Activar modelo**: `POST /models/{model_name}/activate`  
````bash
curl -X POST http://localhost:8000/models/penguins_rf_model_20250824_123000.joblib/activate
````
- **Predecir**: `POST /predict`
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "features": [40.0, 18.0, 190.0, 3800.0]
  }' \
  http://localhost:8000/predict
```

---