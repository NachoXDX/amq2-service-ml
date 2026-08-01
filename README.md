### MLOPS1 - CEIA - FIUBA

## Integrantes

- Rubiolo Pedro Ignacio
- Tolotto Lourdes Sofia
- Bettig Santiago
- Dieguez Manuel

En este repositorio se pretende implementar un servicio de ML para predecir el rendimiento de los estudiantes, utilizando un [dataset de Kaggle](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset/data).

![alt text](diagrama.png)

La implementación incluye:

- MLFlow
  - Se utiliza para llevar un seguimiento de los experimentos, modelos y artefactos. Es usado por distintos servicios del sistema, como Apache Airflow y los Notebooks.

- Apache Airflow
  - Un DAG que obtiene los datos desde google drive, lo verifica, realiza limpieza, escalamiento y encoding y guarda en el bucket los datos separados para entrenamiento y pruebas asi como su procesador de Sklearn. MLflow hace seguimiento de este procesamiento.
  ![alt text](etlDag.png)
  - Un DAG que, dado un nuevo conjunto de datos, reentrena el modelo. Se compara este modelo con el mejor modelo (llamado `champion`), y si es mejor, se reemplaza. Todo se lleva a cabo siendo registrado en MLflow.
  - Un DAG que hace predicciones en bache y las guarda en Valkey para ser servidas al usuario.

- Jupiter Notebooks
  - Un notebook que corre de manera local, que se conecta al data lake para extraer los datos, permitiendo automaticamente seleccionar los mas recientes o un directorio especifico, realiza busqueda de hiperparámetros para una Regresion Logistica con Optuna. Todo el experimento se registra en MLflow, indicando datos utilizados, resultados de cada Trial de Optuna, mejores parametros, metricas y graficas. Además, se registra el modelo en el registro de modelos de MLflow.

- FastAPI
  - Un servicio de API del modelo, que sirve las predicciones del modelo. Intenta buscarlas en cache primero y, si no estan, las calcula utilizando el modelo champion.

- Nginx
  - Un webserver que sirve un frontend para interactuar con el modelo y un reverse proxy para servir el servicio de API de forma mas segura.

- Minio
  - Un data lake donde almacenamos datos crudos, procesados, modelos y artefactos.

- Valkey
  - Un servicio de cache donde almacenamos predicciones que pueden ser servidas al usuario.

- Postgress
  - Utilizado por Apache Airflow y MLFlow para almacenar metadatos.

## Testeo de Funcionamiento

El orden para probar el funcionamiento completo es el siguiente:

1. Levantar todo con docker compose.
2. Ejecutar el DAG de ETL en Airflow llamado `process_etl_student_performance`.
3. Ejecuta la notebook ubicada en `notebooks/student_performance.ipynb` para realizar la búsqueda de hiperparámetros y entrenar el mejor modelo.
4. Ingresar al frontend ubicado en el puerto 8081.
5. Hacer una prediccion completando los campos que solicita el frontend.

## Estado de implementacion

| Componente | Estado |
| :--- | :---: |
| Servicio AirFLow | 🟢 Completo |
| Servicio MLFlow | 🟢 Completo |
| Servicio Minio | 🟢 Completo |
| Servicio Valkey | 🟢 Completo |
| Servicio Postgress | 🟢 Completo |
| Servicio Nginx | 🟢 Completo |
| Servicio FastAPI | 🟢 Completo |
| DAG ETL | 🟢 Completo |
| Notebook Entrenamiento | 🟢 Completo |
| DAG Retrain | 🔴 No implementado |
| DAG Batch Prediction | 🔴 No implementado |
| API | 🔴 No implementado |
| Frontend | 🔴 No implementado |
