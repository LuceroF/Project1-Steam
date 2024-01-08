#Importamos las librerias
import pandas as pd
import numpy as np 
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


#DATA GENERAL DE LA API
app = FastAPI()
app.mount("/static", StaticFiles(directory="image"), name="static")
app.title = "Juegos de Steam - Proyecto Individual N1"
app.description = """
<div style="font-family: Arial, sans-serif; text-align: center;">
    <img src="/static/Banner-deploy.jpg" >
    <p style="font-size: 16px; color: #333;">
        Bienvenidos a la aplicación de recomendación de juegos de Steam.
        Aquí podrás explorar y descubrir juegos a través de nuestras
        <strong>cinco funciones únicas</strong> y <strong>dos modelos de recomendación avanzados</strong>.
    </p>
    <p style="font-size: 14px; color: #555;">
        Esta aplicación te ayudará a encontrar juegos según tus preferencias y te mostrará
        recomendaciones basadas en análisis de datos complejos. ¡Explora y encuentra tu próximo juego favorito!
    </p>
</div>
"""
app.version = "1.0.0"

#DATASETS
funcion_1 = pd.read_parquet('data/funcion_1.parquet')
funcion_2 = pd.read_parquet('data/funcion_2.parquet')
funcion_3 = pd.read_parquet('data/funcion_3.parquet')
funcion_4 = pd.read_parquet('data/funcion_4.parquet')
funcion_5 = pd.read_parquet('data/funcion_5.parquet')
modelo_recomendacion = pd.read_parquet('data/modelo_recomendacion.parquet')

@app.get('/PlayTimeGenre/{genero}', tags=['Consultas sobre Steam Games'])
def PlayTimeGenre(genero: str):
    """
    <font color="blue">
    <h1><u>PlayTimeGenre</u></h1>
    </font>

    <b>Descubre el Año Estrella para tu Género de Juego Favorito.</b><br>
    <b>¿Curioso por saber cuál año fue el más jugado para tu género favorito? Esta función te lo revela.</b><br>

    <em>Parámetros</em><br>
    ----------
    genre : <code>str</code>
    
        Introduce el género de juego en inglés, como "action", "simulation", "indie", etc.

    <em>Resultado</em><br>
    -----------
    Ejemplo:
    ```python
        >>> MostPlayedYearForGenre("action")
    
    {"Año con más horas jugadas para Género Action": 2014}
    ```
    PASOS PARA PROBAR<br>
                            1. Selecciona "Try it out".<br>
                            2. Escribe el género en el campo de texto, en minúsculas o mayúsculas, como prefieras (ejemplo: action).<br>
                            3. Haz clic en "Execute".<br>
                            4. Consulta el "Response body" para ver el año más jugado.
                            </font>
    """

    # Convertimos tanto el género de entrada como la columna del DataFrame a minúsculas
    genero_min = genero.lower()
    funcion_1["genres"] = funcion_1["genres"].str.lower()

    # Filtra los datos para el género especificado
    genre_data = funcion_1[funcion_1["genres"] == genero_min]
    
    # Encuentra el año con más horas jugadas
    if not genre_data.empty:
        max_playtime_row = genre_data.sort_values(by='MaxHours', ascending=False).iloc[0]

        # Devuelve el año de lanzamiento con más horas jugadas para el género dado
        return {"Año de lanzamiento con más horas jugadas para Género " + genero: int(max_playtime_row["year_of_release"])}
    else:
        return {"Error": "Género no encontrado o sin datos"}


def normalize_string(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

@app.get("/UserForGenre/{genre}", tags=['Consultas sobre Steam Games'])
def UserForGenre(genre: str):
    """
    <font color="blue">
    <h1><u>UserForGenre</u></h1>
    </font>

    <b>Explora Quién Domina en tu Género de Juegos Preferido.</b><br>
    <b>Esta función desvela quién ha dedicado más horas a un género de juego en particular, acompañado de un historial detallado por año.</b><br>

    <em>Parámetros</em><br>
    ----------
    genero : <code>str</code>
    
        Escribe el género de juego en inglés, como "action", "simulation", "indie", etc.

    <em>Resultado</em><br>
    -----------
    Ejemplo:
    ```python
        >>> UserForGenre("action")
    
    {"Usuario": "player123", "con más horas jugadas para": "Action", "Historial acumulado": [{"Year": "2012", "Hours Played": 320}] }
    ```
    CÓMO PROBAR<br>
                            1. Haz clic en "Try it out".<br>
                            2. Introduce el género en el campo de texto (ejemplo: action).<br>
                            3. Pulsa "Execute".<br>
                            4. Navega hasta "Response body" para ver los detalles del usuario y su historial.
                            </font>
    """

    # Normalizar el género de entrada
    genero_norm = normalize_string(genre)

    # Normalizar la columna de géneros en el DataFrame y buscar el usuario con más horas jugadas
    funcion_2['genres_norm'] = funcion_2['genres'].apply(normalize_string)
    top_user = funcion_2[funcion_2['genres_norm'] == genero_norm]["user_id"].iloc[0]

    # Filtrar los datos para el usuario y género seleccionado
    user_history = funcion_2[(funcion_2['user_id'] == top_user) & (funcion_2['genres_norm'] == genero_norm)]

    # Seleccionar solo las columnas necesarias y convertir a diccionario
    user_history_filtered = user_history[['Year', 'Hours Played']].copy()
    user_history_dict = user_history_filtered.to_dict(orient="records")

    # Devolver los resultados
    return {"Usuario": top_user, "con más horas jugadas para": genre, "Historial acumulado": user_history_dict}


@app.get('/UsersRecommend/{year}', tags=['Consultas sobre Steam Games'])
def UsersRecommend(year: int):
    """
    <font color="blue">
    <h1><u>UsersRecommend</u></h1>
    </font>

    <b>Los Juegos Más Valorados por los Usuarios para un Año Específico.</b><br>
    <b>Descubre cuáles fueron los top 3 juegos más recomendados por los usuarios en un año concreto.</b><br>

    <em>Parámetros</em><br>
    ----------
    año : <code>int</code>
    
        Año de interés, como 2010, 2012, etc.

    <em>Resultado</em><br>
    -----------
    Ejemplo:
    ```python
        >>> UsersRecommend(2012)
    
    [{"Puesto 1" : "Team Fortress 2"}, {"Puesto 2" : "Terraria"}, {"Puesto 3" : "Garry's Mod"}]
    ```
    CÓMO UTILIZAR<br>
                            1. Haz clic en "Try it out".<br>
                            2. Introduce el Año en el campo de abajo.<br>
                            3. Pulsa "Execute".<br>
                            4. Desplázate hasta "Response body" para descubrir los juegos más valorados.
                            </font>
    """
    # Filtrar reseñas por el año dado, recomendaciones y análisis de sentimiento positivo o neutral
    filtered_reviews = funcion_3[
        (funcion_3['reviews_year'] == year) & 
        (funcion_3['reviews_recommend'] == True) & 
        (funcion_3['sentiment_analysis'].isin([1, 2]))  # 1 para positivo, 2 para neutral
    ]

    # Contar recomendaciones por ID del juego
    recommend_count = filtered_reviews.groupby('reviews_item_id').size().reset_index(name='recommend_count')

    # Ordenar y obtener los 3 juegos principales
    top_3_games = recommend_count.sort_values(by='recommend_count', ascending=False).head(3)

    # Combinar con el dataset de juegos para obtener nombres
    top_3_games_with_names = top_3_games.merge(
        funcion_3[['item_id', 'app_name']].drop_duplicates('item_id'),
        left_on='reviews_item_id',
        right_on='item_id'
    )

    # Preparar el resultado final
    top_3_games_list = top_3_games_with_names[['app_name', 'recommend_count']].to_dict(orient='records')
    top_3_result = []
    for i, game in enumerate(top_3_games_list, start=1):
        game_info = {
            f"Puesto {i}": game['app_name'],
            "Recomendaciones": game['recommend_count']
        }
        top_3_result.append(game_info)

    if not top_3_result:
        return f"No hay suficientes datos para juegos recomendados en el año {year}"
    else:
        response_message = f"Los 3 juegos MÁS recomendados por usuarios para el año {year} son:"
        return {response_message: top_3_result}@app.get('/UsersRecommend/{year}', tags=['Consultas sobre Steam Games'])


@app.get('/UsersWorstDeveloper/{year}', tags=['Consultas sobre Steam Games'])
def UsersWorstDeveloper(year: int):
    """
    <font color="blue">
    <h1><u>UsersWorstDeveloper</u></h1>
    </font>

    <b>Descubre los Desarrolladores de Juegos Menos Favoritos del Año.</b><br>
    <b>Esta función identifica a los tres desarrolladores de juegos que han recibido más reseñas negativas en un año específico.</b><br>

    <em>Parámetros</em><br>
    ----------
    año : <code>int</code>
    
        Año de interés, por ejemplo: 2010, 2012, etc.

    <em>Resultado</em><br>
    -----------
    Ejemplo:
    ```python
        >>> UsersWorstDeveloper(2012)
    
    [{"Puesto 1": "Desarrollador A"}, {"Puesto 2": "Desarrollador B"}, {"Puesto 3": "Desarrollador C"}]
    ```
    INSTRUCCIONES DE USO<br>
                            1. Selecciona "Try it out".<br>
                            2. Introduce el año en el campo de texto correspondiente.<br>
                            3. Haz clic en "Execute".<br>
                            4. Consulta el "Response body" para conocer a los desarrolladores menos favoritos.
                            </font>
    """
    # Filtrar las reseñas para el año dado y donde la recomendación es False
    reviews_year = funcion_4[(funcion_4['reviews_year'] == year) & (funcion_4['reviews_recommend'] == False)]

    # Si no hay reseñas negativas para ese año, retorna mensaje
    if reviews_year.empty:
        return f"No hay datos suficientes."

    # Contar el número de reseñas negativas para cada desarrollador
    developer_negative_reviews = reviews_year['developer'].value_counts()

    # Preparar la salida para los tres primeros desarrolladores, si hay menos de tres, maneja la situación
    top_developers = developer_negative_reviews.head(3)
    formatted_output = [{"Puesto {}".format(i + 1): dev} for i, dev in enumerate(top_developers.index)]
    
    return f"Las 3 desarrolladoras con juegos MENOS recomendados por usuarios para el año {year}: {formatted_output}"

@app.get('/sentiment_analysis/{developer_name}', tags=['Consultas sobre Steam Games'])
def sentiment_analysis(developer_name: str) -> str:
    """
    <font color="blue">
    <h1><u>SentimentAnalysis</u></h1>
    </font>

    <b>Análisis de Sentimiento para Reseñas de Juegos de un Desarrollador Específico.</b><br>
    <b>Explora la recepción general de los juegos de una desarrolladora, desglosando las reseñas en negativas, neutrales y positivas.</b><br>

    <em>Parámetros</em><br>
    ----------
    developer_name : <code>str</code>
    
        Nombre de la empresa desarrolladora. Esta función es insensible a mayúsculas/minúsculas.

    <em>Resultado</em><br>
    -----------
    Ejemplo:
    ```python
        >>> sentiment_analysis("Valve")
    
    "El análisis de sentimiento de valor para la desarrolladora Valve es: [Negative = 10, Neutral = 15, Positive = 75]"
    ```
    INSTRUCCIONES DE USO<br>
                            1. Haz clic en "Try it out".<br>
                            2. Introduce el nombre de la desarrolladora en el campo de texto.<br>
                            3. Haz clic en "Execute".<br>
                            4. Consulta el "Response body" para ver el análisis de sentimiento.
                            </font>
    """
    # Normalizar el nombre de la desarrolladora
    developer_name_normalized = developer_name.lower()

    # Filtrando juegos por la desarrolladora especificada
    filtered_reviews = funcion_5[funcion_5['developer'].str.lower() == developer_name_normalized]

    # Contando las reseñas por análisis de sentimiento
    sentiment_counts = filtered_reviews['sentiment_analysis'].value_counts().sort_index()
    sentiment_counts = { "Negative": sentiment_counts.get(0, 0), 
                         "Neutral": sentiment_counts.get(1, 0), 
                         "Positive": sentiment_counts.get(2, 0) }

    # Formateando el mensaje de salida
    return f"El análisis de sentimiento de reseñas para la desarrolladora {developer_name} es: [Negative = {sentiment_counts['Negative']}, Neutral = {sentiment_counts['Neutral']}, Positive = {sentiment_counts['Positive']}]"

@app.get("/recomendacion_juego/{item_id}", tags=['Sistema de Recomendación'])
def  recomendacion_juego(item_id:int):
    """
    <font color="blue">
    <h1><u>Recomendación de Juegos</u></h1>
    </font>

    <b>Recomendaciones de juegos basadas en un sistema de recomendación item-item.</b><br>
    <b>Obtén una lista de juegos similares basada en un juego específico proporcionado por su ID.</b><br>

    <em>Parámetros</em><br>
    ----------
    item_id : <code>int</code>
    
        El ID numérico del juego. Esta función utiliza el ID para encontrar juegos similares.

    <em>Resultado</em><br>
    -----------
    Ejemplo:
    ```python
        >>> recomendacion_juego(12345)
    
    "Lista de juegos recomendados similares para 12345."
    ```
    INSTRUCCIONES DE USO<br>
                            1. Haz clic en "Try it out".<br>
                            2. Introduce el ID del juego en el campo de texto.<br>
                            3. Haz clic en "Execute".<br>
                            4. Consulta el "Response body" para ver la lista de juegos recomendados.
                            </font>
    """
    
    try:
        game = modelo_recomendacion[modelo_recomendacion['item_id'] == item_id]
    
        if game.empty:
            return("El juego '{item_id}' no posee registros.")
    
        # Obtenemos el índice del juego dado
        idx = game.index[0]

        # Tomamos una muestra aleatoria del DataFrame
        sample_size = 2000
        df_sample = modelo_recomendacion.sample(n=sample_size, random_state=42)
        features_modelo_recomendacion = modelo_recomendacion.select_dtypes(include=[np.number])
        features_df_sample = df_sample.select_dtypes(include=[np.number])

        # Calculamos la similitud de contenido solo para el juego dado y la muestra
        sim_scores = cosine_similarity([features_modelo_recomendacion.iloc[idx]], features_df_sample)

            # Obtenemos las puntuaciones de similitud del juego dado con otros juegos
        sim_scores = sim_scores[0]

        # Ordenamos los juegos por similitud en orden descendente
        similar_games = [(i, sim_scores[i]) for i in range(len(sim_scores)) if i != idx]
        similar_games = sorted(similar_games, key=lambda x: x[1], reverse=True)

        # Obtenemos los 5 juegos más similares
        similar_game_indices = [i[0] for i in similar_games[:5]]

        # Lista de juegos similares (solo nombres)
        similar_game_names = df_sample['item_name'].iloc[similar_game_indices].tolist()

        # Formateamos el mensaje de salida
        respuesta = {
            "Lista de juegos recomendados similares para": game['item_name'].iloc[0],
            "Juegos Recomendados": similar_game_names
        }

        return respuesta
    except Exception as e:
        # Manejo genérico de excepciones, puedes especificar errores más específicos si lo prefieres
        raise HTTPException(status_code=500, detail="El ID del juego debe ser un número entero.")
