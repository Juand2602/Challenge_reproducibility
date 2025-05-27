# README - Evolution Strategies con MLflow

Este proyecto implementa un algoritmo de **Evolution Strategies (ES)** aplicado al entorno `CartPole` de OpenAI Gym, utilizando **MLflow** para el registro de parámetros, métricas y artefactos.

---
## Descripción del Experimento

* Entorno: `CartPole`
* Modelo: MLP con dos capas ocultas
* Algoritmo: Evolution Strategies (ES)
* Registro con: MLflow

Se entrenaron **3 versiones** del algoritmo modificando los hiperparámetros:

| Versión | Pop Size | Sigma | Alpha |
| ------- | -------- | ----- | ----- |
| V1      | 50       | 0.1   | 0.01  |
| V2      | 100      | 0.2   | 0.02  |
| V3      | 150      | 0.3   | 0.03  |

**Nota:** Estos valores son de ejemplo. Ver resultados reales en la UI de MLflow.

---

## Hiperparámetros Registrados

* `pop_size`
* `sigma`
* `alpha`
* `iterations`
* `env_name`

## Métricas Registradas

* `test_reward` (por iteración)

## Artefactos Registrados

* Pesos del modelo (`CartPole.npy`)
* Recompensas por iteración (`CartpPoleRewards.npy`)
* Gráfica de entrenamiento (`CartPole.png`)
* Video del agente (`CartPole.mp4`)

---

## Modelo Seleccionado

* Nombre en el registro: `ES_Swimmer_Model`
* Versión: 1
* Etapa: `Staging`

---

## Cómo Cargar el Modelo

```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/ES_Swimmer_Model/Staging")
```

---

## Estructura del Proyecto

```
/home/hadoop/MLFLOW_taller/
├── swimmer_es_mlflow.py
├── README.md
├── results/
│   ├── paramsSwimmer-v4Para.npy
│   └── rewardsSwimmer-v4Para.npy
├── Swimmer-v4Para.png
├── Swimmer-v4Para.mp4
└── mlruns/ 
```

---

## Cómo Ejecutar

```bash
python swimmer_es_mlflow.py
mlflow ui
# Abrir en navegador: http://127.0.0.1:5000
```

---

Para más detalles, consulta el script `swimmer_es_mlflow.py`.
