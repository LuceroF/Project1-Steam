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
app.title = "Steam Games - Individual Project N1"
app.description = """
<div style="font-family: Arial, sans-serif; text-align: center;">
    <img src="/static/Banner-deploy.jpg" alt="Application Banner">
    <p style="font-size: 16px; color: #333;">
        Welcome to the Steam game recommendation application.
        Here you can explore and discover games through our
        <strong>five unique features</strong> and <strong>two advanced recommendation models</strong>.
    </p>
    <p style="font-size: 14px; color: #555;">
        This application will help you find games according to your preferences and will show you
        recommendations based on complex data analysis. Explore and find your next favorite game!
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

@app.get('/PlayTimeGenre/{genero}', tags=['Inquiries about Steam Games'])
def PlayTimeGenre(genero: str):
    """
    <font color="blue">
    <h1><u>PlayTimeGenre</u></h1>
    </font>

    <b>Discover the Star Year for Your Favorite Game Genre.</b><br>
    <b>Curious to know which year was the most played for your favorite genre? This function reveals it to you.</b><br>

    <em>Parameters</em><br>
    ----------
    genre : <code>str</code>
    
        Enter the game genre in English, such as "action", "simulation", "indie", etc.

    <em>Result</em><br>
    -----------
    Example:
    ```python
        >>> MostPlayedYearForGenre("action")
    
    {"Year with Most Playtime for Action Genre": 2014}
    ```
    STEPS TO TEST<br>
                            1. Select "Try it out".<br>
                            2. Type the genre in the text field, in lowercase or uppercase, as you prefer (example: action).<br>
                            3. Click on "Execute".<br>
                            4. Check the "Response body" to see the most played year.
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
        return {"Year of Release with Most Playtime for Genre " + genero: int(max_playtime_row["year_of_release"])}
    else:
        return {"Error": "Genre not found or without data"}


def normalize_string(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

@app.get("/UserForGenre/{genre}", tags=['Inquiries about Steam Games'])
def UserForGenre(genre: str):
    """
    <font color="blue">
    <h1><u>UserForGenre</u></h1>
    </font>

    <b>Explore Who Dominates in Your Favorite Game Genre.</b><br>
    <b>This function reveals who has dedicated the most hours to a particular game genre, accompanied by a detailed yearly history.</b><br>

    <em>Parameters</em><br>
    ----------
    genre : <code>str</code>

        Enter the game genre in English, such as "action", "simulation", "indie", etc.

    <em>Result</em><br>
    -----------
    Example:
    ```python
        >>> UserForGenre("action")

    {"User": "player123", "with most hours played for": "Action", "Accumulated History": [{"Year": "2012", "Hours Played": 320}] }
    HOW TO TEST<br>
                    1. Click on "Try it out".<br>
                    2. Enter the genre in the text field (example: action).<br>
                    3. Press "Execute".<br>
                    4. Navigate to "Response body" to see the user details and their history.
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


@app.get('/UsersRecommend/{year}', tags=['Inquiries about Steam Games'])
def UsersRecommend(year: int):
    """
    <font color="blue">
    <h1><u>UsersRecommend</u></h1>
    </font>

    <b>The Most Highly Rated Games by Users for a Specific Year.</b><br>
    <b>Discover the top 3 games most recommended by users in a particular year.</b><br>

    <em>Parameters</em><br>
    ----------
    year : <code>int</code>
    
        The year of interest, such as 2010, 2012, etc.

    <em>Result</em><br>
    -----------
    Example:
    ```python
        >>> UsersRecommend(2012)
    
    [{"1st Place" : "Team Fortress 2"}, {"2nd Place" : "Terraria"}, {"3rd Place" : "Garry's Mod"}]
    ```
    HOW TO USE<br>
                        1. Click on "Try it out".<br>
                        2. Enter the Year in the field below.<br>
                        3. Press "Execute".<br>
                        4. Scroll to "Response body" to find out the most highly rated games.
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
            f"Position {i}": game['app_name'],
            "Recomendations": game['recommend_count']
        }
        top_3_result.append(game_info)

    if not top_3_result:
        return f"Not enough data for recommended games in the year {year}"
    else:
        response_message = f"The top 3 MOST recommended games by users for the year {year} are:"
        return {response_message: top_3_result}


@app.get('/UsersWorstDeveloper/{year}', tags=['Inquiries about Steam Games'])
def UsersWorstDeveloper(year: int):
    """
    <font color="blue">
    <h1><u>UsersWorstDeveloper</u></h1>
    </font>

    <b>Discover the Least Favorite Game Developers of the Year.</b><br>
    <b>This function identifies the three game developers who have received the most negative reviews in a specific year.</b><br>

    <em>Parameters</em><br>
    ----------
    year : <code>int</code>
    
        Year of interest, for example: 2010, 2012, etc.

    <em>Result</em><br>
    -----------
    Example:
    ```python
        >>> UsersWorstDeveloper(2012)
    
    [{"1st Place": "Developer A"}, {"2nd Place": "Developer B"}, {"3rd Place": "Developer C"}]
    ```
    USAGE INSTRUCTIONS<br>
                            1. Select "Try it out".<br>
                            2. Enter the year in the corresponding text field.<br>
                            3. Click on "Execute".<br>
                            4. Check the "Response body" to find out the least favorite developers.
    </font>
    """

    # Filtrar las reseñas para el año dado y donde la recomendación es False
    reviews_year = funcion_4[(funcion_4['reviews_year'] == year) & (funcion_4['reviews_recommend'] == False)]

    # Si no hay reseñas negativas para ese año, retorna mensaje
    if reviews_year.empty:
        return f"Not enough data available."

    # Contar el número de reseñas negativas para cada desarrollador
    developer_negative_reviews = reviews_year['developer'].value_counts()

    # Preparar la salida para los tres primeros desarrolladores, si hay menos de tres, maneja la situación
    top_developers = developer_negative_reviews.head(3)
    formatted_output = [{"Position {}".format(i + 1): dev} for i, dev in enumerate(top_developers.index)]
    
    return f"The top 3 developers with the LEAST recommended games by users for the year {year}: {formatted_output}"

@app.get('/sentiment_analysis/{developer_name}', tags=['Inquiries about Steam Games'])
def sentiment_analysis(developer_name: str) -> str:
    """
    <font color="blue">
    <h1><u>SentimentAnalysis</u></h1>
    </font>

    <b>Sentiment Analysis for Game Reviews of a Specific Developer.</b><br>
    <b>Explore the overall reception of a developer's games, breaking down the reviews into negative, neutral, and positive.</b><br>

    <em>Parameters</em><br>
    ----------
    developer_name : <code>str</code>
    
        Name of the developer company. This function is case insensitive.

    <em>Result</em><br>
    -----------
    Example:
    ```python
        >>> sentiment_analysis("Valve")
    
    "The sentiment analysis value for developer Valve is: [Negative = 10, Neutral = 15, Positive = 75]"
    ```
    USAGE INSTRUCTIONS<br>
                            1. Click on "Try it out".<br>
                            2. Enter the developer's name in the text field.<br>
                            3. Click on "Execute".<br>
                            4. Check the "Response body" for the sentiment analysis.
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
    return f"The sentiment analysis of reviews for the developer {developer_name} is: [Negative = {sentiment_counts['Negative']}, Neutral = {sentiment_counts['Neutral']}, Positive = {sentiment_counts['Positive']}]"

modelo_recomendacion = modelo_recomendacion.drop_duplicates(subset='item_id')

@app.get("/recomendacion_juego/{item_id}", tags=['Recommendation System'])
def  recomendacion_juego(item_id:int):
    """
    <font color="blue">
    <h1><u>Game Recommendation</u></h1>
    </font>

    <b>Game recommendations based on an item-item recommendation system.</b><br>
    <b>Get a list of similar games based on a specific game provided by its ID.</b><br>

    <em>Parameters</em><br>
    ----------
    item_id : <code>int</code>
    
        The numeric ID of the game. This function uses the ID to find similar games.

    <em>Result</em><br>
    -----------
    Example:
    ```python
        >>> game_recommendation(12345)
    
    "List of similar recommended games for 12345."
    ```
    USAGE INSTRUCTIONS<br>
                            1. Click on "Try it out".<br>
                            2. Enter the game ID in the text field.<br>
                            3. Click on "Execute".<br>
                            4. Check the "Response body" to see the list of recommended games.
    </font>
    """

    
    try:
        game = modelo_recomendacion[modelo_recomendacion['item_id'] == item_id]
    
        if game.empty:
            return f"The game '{item_id}' does not have any records."
    
        # Obtenemos el índice del juego dado
        idx = game.index[0]

        # Tomamos una muestra aleatoria del DataFrame
        sample_size = 2000
        df_sample = modelo_recomendacion[modelo_recomendacion['item_id'] != item_id].sample(n=sample_size, random_state=42)
        features_modelo_recomendacion = modelo_recomendacion.select_dtypes(include=[np.number])
        features_df_sample = df_sample.select_dtypes(include=[np.number])

        # Calculamos la similitud de contenido solo para el juego dado y la muestra
        sim_scores = cosine_similarity([features_modelo_recomendacion.iloc[idx]], features_df_sample)

        # Obtenemos las puntuaciones de similitud del juego dado con otros juegos
        sim_scores = sim_scores[0]

        # Ordenamos los juegos por similitud en orden descendente
        similar_games = [(i, sim_scores[i]) for i in range(len(sim_scores)) if i != idx]
        similar_games = sorted(similar_games, key=lambda x: x[1], reverse=True)

    # Obtener los 5 juegos más similares asegurándose de que sean únicos
        similar_game_indices = [i[0] for i in similar_games[:5]]
        unique_similar_games = list(dict.fromkeys(df_sample['item_name'].iloc[similar_game_indices]))

        # Formatear el mensaje de salida para que solo incluya juegos únicos
        respuesta = {
        "List of similar recommended games for": game['item_name'].iloc[0],
        "Recommended Games": unique_similar_games
        }

        return respuesta
    except Exception as e:
        # Manejo genérico de excepciones, puedes especificar errores más específicos si lo prefieres
        raise HTTPException(status_code=500, detail="The game ID must be an integer number.")
