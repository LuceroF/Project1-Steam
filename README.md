![Presentacion](image/Banner-deploy.jpg)
# Proyecto de Recomendación de Videojuegos en Steam

## Descripción General
En el corazón de Steam, este proyecto surge para potenciar la experiencia de juego con un **sistema de recomendación personalizado**, diseñado para adaptarse y evolucionar según las preferencias de nuestros usuarios. Como Data Scientist, he fusionado técnicas de **Machine Learning** para no solo afinar las recomendaciones, sino también para destilar análisis profundos del comportamiento de juego. Estos análisis son cruciales, proporcionando una base de datos rica para el desarrollo estratégico de productos y actualizaciones. El objetivo es mantener a Steam como líder innovador, garantizando que cada jugador encuentre su próximo juego favorito con facilidad y que cada título tenga la oportunidad de brillar.

## Tecnologías y Herramientas Utilizadas
- **Lenguaje de Programación:** Python
- **Librerías y Frameworks Principales:**
  - FastAPI para el desarrollo de la API
  - NLTK para el análisis de sentimientos
  - _Otras librerías incluyen matplotlib, pandas, numpy, seaborn, etc._

## Proceso de Desarrollo
## ETL

El proceso de ETL se realizó utilizando tres bases de datos principales en formato gzip: `user_items`, `user_reviews` y `steam_games`. Se efectuó una limpieza y transformación de estos datos, almacenándolos en formato parquet para optimizar el rendimiento en la API y el modelo de ML. (Vea el archivo adjunto: [ETL.ipynb](/ETL.ipynb) para más detalles).

## Entrenamiento y Mantenimiento del Modelo ML

El modelo de recomendación se basa en un enfoque item-item. Se seleccionaron características relevantes y se utilizó la similitud de coseno para generar recomendaciones de juegos similares. (Vea el archivo adjunto: [Funciones.ipynb](/Funciones.ipynb) para más detalles).

## Características del Modelo de ML
El sistema de recomendación `item-item` se implementa mediante el endpoint `/recomendacion_juego/{item_id}`, que retorna una lista de 5 juegos recomendados similares al juego de entrada basado en su ID.

## API y Endpoints
La API, desarrollada con FastAPI, expone varios endpoints para acceder a la información y funcionalidades del sistema:

1. **PlayTimeGenre**
2. **UserForGenre**
3. **UsersRecommend**
4. **UsersWorstDeveloper**
5. **sentiment_analysis**

La implementación de la API y los endpoints se describe en detalle en el archivo main.py. (Consulte [main.py](/main.py) para más información).

## Despliegue

El despliegue se realizó en [Render.com](https://render.com/), conectando el repositorio de GitHub con la plataforma para un despliegue continuo y eficiente. Puedes observarlo aquí: [steam-mlops-utyg.onrender.com](https://steam-mlops-utyg.onrender.com/docs).

## Análisis Exploratorio de Datos (EDA)
El EDA proporcionó insights valiosos sobre las relaciones entre las variables, patrones y posibles anomalías en los datos. (Vea el archivo adjunto: [EDA.ipynb](/EDA.ipynb) para más detalles).

## Puntos Destacados del Proyecto
- La base del proyecto es una **API RESTful** dinámica, diseñada para ser universalmente accesible.
- La implementación se realizó utilizando **FastAPI**, una elección estratégica para garantizar un desarrollo ágil y estable de la API.
- El despliegue se efectuó en **Render**, seleccionado por su eficiencia y sencillez en la integración continua.
- La fase de **análisis exploratorio de datos** se abordó manualmente, proporcionando una comprensión detallada y un conocimiento profundo de nuestros conjuntos de datos.

## Reflexiones Finales y Perspectivas de Futuro

En este proyecto, se ha demostrado cómo las disciplinas de **Data Engineering** y **Machine Learning** pueden aplicarse para resolver problemas reales en el ámbito empresarial de los videojuegos en **Steam**. Se ha documentado cada paso y recurso utilizado, facilitando la replicación y el entendimiento integral del enfoque propuesto. Todo el proceso y los resultados se encuentran disponibles en el **repositorio detallado** que se ha creado.

El proyecto también ha permitido explorar las posibilidades de optimización y avance con recursos adicionales en **Render**. Se ha logrado desarrollar un sistema eficiente y ágil, que se ha desplegado en **Render.com** con una interfaz intuitiva y una documentación completa. Estas acciones han mejorado la experiencia del usuario y han aportado lecciones valiosas sobre la implementación práctica de sistemas de ingeniería y aprendizaje automático en entornos de producción reales.

Este proyecto ha sido un desafío y una oportunidad para crecer profesionalmente y contribuir al ámbito de los videojuegos en **Steam**.
