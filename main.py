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
steam_games = pd.read_csv('data/steam_games_cleaned.csv')
user_reviews = pd.read_csv('data/user_reviews_cleaned.csv')
user_items = pd.read_parquet('data/user_items_cleaned.parquet')
genre= steam_games[["item_id","genres","year_of_release"]]

# Función auxiliar para normalizar cadenas de texto
def normalize_string(s):
    return s.lower().strip()

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
    # Preparar datos
    user_items['user_id'] = user_items['user_id'].astype('string')
    play_time_user = user_items[["item_id", "playtime_forever", "user_id"]]
    play_time_user['item_id'] = play_time_user['item_id'].astype('float64')
    genre_playtime = play_time_user.merge(genre, on="item_id")
    combined_genre_user = play_time_user.merge(genre, on="item_id")

    # Convertir a string y agrupar
    combined_genre_user = combined_genre_user.astype({'genres': 'string', 'year_of_release': 'string', 'user_id': 'string'})
    total_playtime_by_user_year = combined_genre_user.groupby(["genres", "user_id", "year_of_release"])["playtime_forever"].sum().reset_index()
    total_hours_by_user = combined_genre_user.groupby(["genres", "user_id"])["playtime_forever"].sum().reset_index()
    max_hours_by_genre = total_hours_by_user.groupby("genres")["playtime_forever"].agg(playtime_forever="max").reset_index()

    # Combinar y preparar DataFrame
    detailed_max_hours = pd.merge(max_hours_by_genre, total_hours_by_user, on=["genres", "playtime_forever"])
    combined_user_playtime = pd.merge(detailed_max_hours, total_playtime_by_user_year.rename(columns={"playtime_forever": "playtime_forever_year"}))

    # Normalización y búsqueda de usuario
    genero_norm = normalize_string(genero)
    combined_user_playtime['genres_norm'] = combined_user_playtime['genres'].apply(normalize_string)
    user_genre_data = combined_user_playtime[combined_user_playtime['genres_norm'] == genero_norm]

    if user_genre_data.empty:
        return {"Error": "Género no encontrado o sin datos. Por favor, ingresa un género válido."}

    top_user = user_genre_data["user_id"].iloc[0]
    user_history = user_genre_data[user_genre_data['user_id'] == top_user][['year_of_release', 'playtime_forever_year']].rename(columns={'year_of_release': 'Year', 'playtime_forever_year': 'Hours Played'})
    user_history_dict = user_history.to_dict(orient="records")

    # Devolver los resultados
    return {"Usuario": top_user, "con más horas jugadas para": genero, "Historial acumulado": user_history_dict}

from fastapi import FastAPI
import pandas as pd

app = FastAPI()

# Función auxiliar para normalizar cadenas de texto
def normalize_string(s):
    return s.lower().strip()

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
    # Reutilizando la preparación de datos de UserForGenre
    user_items['user_id'] = user_items['user_id'].astype('string')
    play_time_user = user_items[["item_id", "playtime_forever", "user_id"]]
    play_time_user['item_id'] = play_time_user['item_id'].astype('float64')
    combined_genre_user = play_time_user.merge(genre, on="item_id")
    combined_genre_user = combined_genre_user.astype({'genres': 'string', 'year_of_release': 'string', 'user_id': 'string'})

    # Normalización y búsqueda del año con más horas jugadas
    genero_norm = normalize_string(genre)
    combined_genre_user['genres_norm'] = combined_genre_user['genres'].apply(normalize_string)
    genre_year_data = combined_genre_user[combined_genre_user['genres_norm'] == genero_norm]

    if genre_year_data.empty:
        return {"Error": "Género no encontrado o sin datos. Por favor, ingresa un género válido."}

    most_played_year = genre_year_data.groupby("year_of_release")["playtime_forever"].sum().idxmax()

    # Devolver los resultados
    return {"Año con más horas jugadas para Género": genre, "Año": most_played_year}


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
    # Filtra las reseñas para el año dado, recomendaciones positivas o neutrales
    filtered_reviews = user_reviews[
        (user_reviews['year'] == year) & 
        (user_reviews['reviews_recommend'] == True) & 
        (user_reviews['sentiment_analysis'].isin([1, 2]))  # 1 para positivo, 2 para neutral
    ]

    # Contar recomendaciones por ID de juego
    recommend_count = filtered_reviews.groupby('reviews_item_id').size().reset_index(name='recommend_count')

    # Ordenar y obtener los 3 juegos principales
    top_3_games = recommend_count.sort_values(by='recommend_count', ascending=False).head(3)

    # Fusionar con el conjunto de datos de juegos para obtener nombres
    top_3_games_with_names = pd.merge(top_3_games, steam_games.drop_duplicates(subset='item_id'), 
                                      left_on='reviews_item_id', right_on='item_id')

    # Preparar el resultado final
    top_3_games_list = top_3_games_with_names[['app_name', 'recommend_count']].to_dict(orient='records')
    top_3_result = [{"Puesto 1": top_3_games_list[0]['app_name']}, 
                    {"Puesto 2": top_3_games_list[1]['app_name']}, 
                    {"Puesto 3": top_3_games_list[2]['app_name']}]

    return f"Los 3 juegos más recomendados por usuarios para el año {year} son: {top_3_result}"


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
    reviews_year = user_reviews[(user_reviews['year'] == year) & (user_reviews['reviews_recommend'] == False)]

    # Combinar las reseñas con los datos de los juegos para obtener la información del desarrollador
    merged_data = reviews_year.merge(steam_games, left_on='reviews_item_id', right_on='item_id')

    # Contar el número de reseñas negativas para cada desarrollador para el año dado
    developer_negative_reviews = merged_data['developer'].value_counts()

    # Obtener los tres principales desarrolladores con más reseñas negativas
    top_3_developers = developer_negative_reviews.head(3)

    # Formatear la salida
    formatted_output = [{"Puesto 1": top_3_developers.index[0]}, 
                        {"Puesto 2": top_3_developers.index[1]}, 
                        {"Puesto 3": top_3_developers.index[2]}]
    
    # Construyendo el mensaje
    message = f"Las 3 desarrolladoras con juegos MENOS recomendados por usuarios para el año {year}:"

    return message, formatted_output

@app.get('/sentiment_analysis/{developer_name}')
def sentiment_analysis(developer_name: str) -> str:
    # Normalizar el nombre de la desarrolladora para hacerlo insensible a mayúsculas/minúsculas
    developer_name_normalized = developer_name.lower()

    # Filtrando juegos por la desarrolladora especificada
    filtered_games = steam_games[steam_games['developer'].str.lower() == developer_name_normalized]

    # Obteniendo los IDs de los juegos de la desarrolladora
    game_ids = filtered_games['item_id'].unique()

    # Filtrando las reseñas para los juegos de la desarrolladora
    reviews = user_reviews[user_reviews['reviews_item_id'].isin(game_ids)]

    # Contando las reseñas por análisis de sentimiento
    sentiment_counts = reviews['sentiment_analysis'].value_counts().sort_index()
    sentiment_counts = { "Negative": sentiment_counts.get(0, 0), 
                         "Neutral": sentiment_counts.get(1, 0), 
                         "Positive": sentiment_counts.get(2, 0) }

    # Formateando el mensaje de salida
    return f"El análisis de sentimiento de valor para la desarrolladora {developer_name} es: [Negative = {sentiment_counts['Negative']}, Neutral = {sentiment_counts['Neutral']}, Positive = {sentiment_counts['Positive']}]"
