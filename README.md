# Challenge de Reproducibilidad: Estrategias Evolutivas como Alternativa Escalable al Aprendizaje por Refuerzo

## Integrantes del Equipo

* Daniver Franchesco Hernandez Acero 
* Jose Miguel Pardo Diaz
* Juan Diego Sepulveda Herrera

## Descripción del Challenge

Este repositorio contiene el desarrollo del challenge de reproducibilidad para la materia de **Análisis de Datos a Gran Escala**. El objetivo principal es seleccionar un artículo científico que presente metodologías escalables, y reproducir desde cero su implementación. Este proceso busca evaluar la capacidad del equipo para comprender, documentar y replicar procesos complejos de análisis de datos de manera precisa y transparente, fomentando buenas prácticas en programación reproducible, control de versiones y validación de resultados.

## Artículo Científico Seleccionado

El artículo seleccionado para este challenge es:

**Título:** Evolution Strategies as a Scalable Alternative to Reinforcement Learning

**Autores:** Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever

**Publicación:** arXiv:1703.03864v2 [stat.ML] 7 Sep 2017

**Resumen:** El artículo explora el uso de Estrategias Evolutivas (ES), una clase de algoritmos de optimización de caja negra, como una alternativa a técnicas populares de Aprendizaje por Refuerzo (RL) basadas en MDP como Q-learning y Gradientes de Política. Los experimentos en MuJoCo y Atari demuestran que ES es una estrategia viable que escala bien con el número de CPUs, permitiendo resolver tareas complejas como la caminata humanoide 3D en minutos y obtener resultados competitivos en Atari.

## Objetivos del Proyecto

* Comprender en profundidad la metodología de Estrategias Evolutivas propuesta en el paper.
* Implementar desde cero el algoritmo ES y sus variantes para diferentes entornos.
* Reproducir los experimentos clave presentados en el artículo en entornos MuJoCo y Atari.
* Comparar los resultados obtenidos con los publicados en el paper original.
* (Opcional/Extensión) Comparar el rendimiento de ES con otros algoritmos de RL como PPO en entornos seleccionados.

## Metodología Implementada (Resumen)

Este proyecto incluye las siguientes implementaciones principales:

1.  **Estrategias Evolutivas (ES) para MuJoCo:**
    * Implementación del algoritmo ES utilizando **NumPy**.
    * Política de Perceptrón Multicapa (MLP) con dos capas ocultas y activaciones `tanh`.
    * Normalización de recompensas (z-score) para la actualización de parámetros.
    * Versiones secuenciales y paralelizadas (utilizando `multiprocessing`) del algoritmo.
2.  **Estrategias Evolutivas (ES) para Atari (Pong):**
    * Implementación del algoritmo ES utilizando **PyTorch**.
    * Política de Red Neuronal Convolucional (CNN) basada en la arquitectura Nature DQN/A3C.
    * Preprocesamiento de imágenes estándar para Atari (escala de grises, redimensionamiento, apilamiento de frames).
    * Normalización de recompensas (z-score) para la actualización de parámetros.
3.  **Proximal Policy Optimization (PPO) (Comparación):**
    * Uso de la implementación de PPO de la biblioteca **Stable Baselines3**.
    * Entrenamiento de agentes PPO en entornos MuJoCo seleccionados para comparar el rendimiento con ES.

## Entornos Probados

* **MuJoCo (con ES en NumPy):**
    * `InvertedPendulum-v4`
    * `HalfCheetah-v4`
    * `Hopper-v4`
    * `Swimmer-v4`
    * `Walker2d-v4`
* **Atari (con ES en PyTorch):**
    * `ALE/Pong-v5`
* **MuJoCo (con PPO):**
    ** `InvertedPendulum-v4`
    * `HalfCheetah-v4`
    * `Hopper-v4`
    * `Swimmer-v4`
    * `Walker2d-v4`
## Enlace al video

[![Challenge de Reproducibilidad]](https://youtu.be/S54cFXPm528)



