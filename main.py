#Importamos las librerias
import pandas as pd
import numpy as np 
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI
import unicodedata
app = FastAPI()

#DATA GENERAL DE LA API
app = FastAPI()
app.title = "Steam API - ML SteamGame"
app.version = "1.0.0"

#DATASETS
funcion_1 = pd.read_parquet('data/funcion_1.parquet')
funcion_2 = pd.read_parquet('data/funcion_2.parquet')
funcion_3 = pd.read_parquet('data/funcion_3.parquet')
funcion_4 = pd.read_parquet('data/funcion_4.parquet')
funcion_5 = pd.read_parquet('data/funcion_5.parquet')

@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):
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

@app.get("/MostPlayedYearForGenre/{genre}")
def MostPlayedYearForGenre(genre: str):
    """
    <font color="blue">
    <h1><u>MostPlayedYearForGenre</u></h1>
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


@app.get('/UsersRecommend/{year}')
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
        funcion_3[['item_id', 'title']].drop_duplicates('item_id'),
        left_on='reviews_item_id',
        right_on='item_id'
    )

    # Preparar el resultado final
    top_3_games_list = top_3_games_with_names[['title', 'recommend_count']].to_dict(orient='records')
    top_3_result = []
    for i in range(min(3, len(top_3_games_list))):
        game_info = {
            "Juego": top_3_games_list[i]['title'],
            "Recomendaciones": top_3_games_list[i]['recommend_count']
        }
        top_3_result.append(game_info)

    if not top_3_result:
        return f"No hay suficientes datos para juegos recomendados en el año {year}"
    else:
        return top_3_result


@app.get('/UsersWorstDeveloper/{year}')
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

@app.get('/sentiment_analysis/{developer_name}')
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