from fastapi import FastAPI
import pandas as pd
from fastapi.responses import JSONResponse

#DATA GENERAL DE LA API
app = FastAPI()
app.title = "Steam API - ML SteamGameRecommender"
app.version = "1.0.0"


#DATASETS

df_most_played_year = pd.read_csv("./Data_API/df_most_played_year.csv")
df_maxuser_genre_cleaned = pd.read_csv('./Data_API/df_maxuser_genre_cleaned.csv')
df_top_positive_games_by_year = pd.read_csv('./Data_API/df_top_positive_games_by_year.csv')
df_top_negative_games_by_year = pd.read_csv('./Data_API/df_top_negative_games_by_year.csv')
df_sentiment_counts = pd.read_csv('./Data_API/df_sentiment_counts.csv')


# Endpoint 1

# Ruta para obtener el año con más horas jugadas para un género específico
@app.get("/most_played_genre/{genre}")
async def PlayTimeGenre(genre):
    """
    <font color="blue">
    <h1><u>PlayTimeGenre</u></h1>
    </font>

    <b>Año con más horas jugadas para el género dado.</b><br>
    <b>Esta función recibe un género y retorna el año de lanzamiento con más horas jugadas para ese género.</b><br>

    <em>Parámetros</em><br>
    ----------
    genre : <code>str</code>
    
        Género en inglés, por ejemplo: "Action", "Simulation", "Indie", etc.

    <em>Retorno</em><br>
    -----------
    Ejemplo:
    ```python
        >>> PlayTimeGenre("Action")
    
    {"Año de lanzamiento con más horas jugadas para Género Action" : 2012}
    ```
    INSTRUCCIONES<br>
                        1. Haga clic en "Try it out".<br>
                        2. Ingrese el Género en el cuadro de abajo. (Primera letra Mayuscula, ejemplo: Indie)<br>
                        3. Presiona Excute.<br>
                        4. Desplácese hacia "Response body" para ver el resultado.
                        </font>

    """
    # Filtra el DataFrame por el género especificado
    genre_data = df_most_played_year[df_most_played_year['genres'] == genre]
    
    if genre_data.empty:
        return f'El género {genre} no se encuentra en el conjunto de datos.'
    
    # Obtiene el año directamente y conviértelo a un tipo de dato int
    max_playtime_year = int(genre_data.iloc[0]['release_year'])
    
    # Devuelve el resultado en el formato deseado
    result = {f"Año de lanzamiento con más horas jugadas para el género {genre}": max_playtime_year}
    
    return result



#Endpoint 2
@app.get("/most_hours_played_genre/{genre}")
def UserForGenre(genre):
    """
        <font color="blue">
        <h1><u>UserForGenre</u></h1>
        </font>

        <b>Usuario con más horas jugadas para Género dado.</b><br>
        <b>Esta función recibe un género y retorna  el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.</b><br>

        <em>Parámetros</em><br>
        ----------
        genre : <code>str</code>
        
            Género en inglés, por ejemplo: "Action", "Simulation", "Indie", etc.

        <em>Retorno</em><br>
        -----------
        Ejemplo:
        ```python
            >>> UserForGenre("Action")
        
        {"Usuario con más horas jugadas para Género Action" : "thiefofrosesinlalaland, "Horas jugadas":
        [{{Año: 2014, Horas: 7107.58}, {Año: 2015, Horas: 331.12}]}
        ```
        INSTRUCCIONES<br>
                            1. Haga clic en "Try it out".<br>
                            2. Ingrese el Género en el cuadro de abajo. (Primera letra Mayuscula, ejemplo: Simulation)<br>
                            3. Presiona Excute.<br>
                            4. Desplácese hacia "Response body" para ver el resultado.
                            </font>

        """
    # Filtra el DataFrame por el género especificado
    genre_data = df_maxuser_genre_cleaned[df_maxuser_genre_cleaned['genres'] == genre]
    
    if genre_data.empty:
        return f'El género {genre} no se encuentra en el conjunto de datos.'
    
    # Encuentra el usuario con más horas jugadas para el género
    max_playtime_user = genre_data['user_id'].values[0]
    
    # Crea una lista de acumulación de horas jugadas por año
    playtime_by_year = []
    for column in genre_data.columns[3:]:  # Ignora las primeras tres columnas
        year = int(column)
        playtime = round(genre_data[column].sum() , 2)  # Convierte minutos a horas y redondea a 2 decimales
        if playtime > 0:
            playtime_by_year.append({'Año': year, 'Horas': playtime})
    
    # Devuelve el resultado en el formato deseado
    result = {
        f"Usuario con más horas jugadas para Género {genre}": max_playtime_user,
        "Horas jugadas": playtime_by_year
    }

    return result

#3
@app.get("/top_3_MOST_recommended_games/{year}")
def UsersRecommend(year):
    """
        <font color="blue">
        <h1><u>UsersRecommend</u></h1>
        </font>

        <b>Top 3 de juegos MÁS recomendados por usuarios para el año dado. .</b><br>
        <b>Esta función recibe un año y retorna el top 3 de juegos MÁS recomendados por usuarios para el año dado. .</b><br>

        <em>Parámetros</em><br>
        ----------
        año : <code>int</code>
        
            Año, por ejemplo: 2010, 2012, etc.

        <em>Retorno</em><br>
        -----------
        Ejemplo:
        ```python
            >>> UsersRecommend(2012)
        
        [{"Puesto 1" : "Team Fortress 2"}, {"Puesto 2" : "Terraria"},{"Puesto 3" : "Garry's Mod"}]
        
        ```
        INSTRUCCIONES<br>
                            1. Haga clic en "Try it out".<br>
                            2. Ingrese el Año en el cuadro de abajo. <br>
                            3. Presiona Excute.<br>
                            4. Desplácese hacia "Response body" para ver el resultado.
                            </font>

        """
    # Filtrar el DataFrame por el año dado
    filtered_df = df_top_positive_games_by_year[df_top_positive_games_by_year['reviews_year'] == year]
    
    # Verificar si se encontraron revisiones para el año dado
    if filtered_df.empty:
        return "Año ingresado inválido o sin revisiones."
    
    top_games_list = [
        {"Puesto 1": filtered_df.iloc[0]['Rank 1']},
        {"Puesto 2": filtered_df.iloc[0]['Rank 2']},
        {"Puesto 3": filtered_df.iloc[0]['Rank 3']}
    ]
    return top_games_list

#4
@app.get("/top_3_LEAST_recommended_games/{year}")
def UsersNotRecommend(year):
    """
        <font color="blue">
        <h1><u>sersNotRecommend</u></h1>
        </font>

        <b>Top 3 de juegos MENOS recomendados por usuarios para el año dado. .</b><br>
        <b>Esta función recibe un año y retorna el top 3 de juegos MENOS recomendados por usuarios para el año dado. .</b><br>

        <em>Parámetros</em><br>
        ----------
        año : <code>int</code>
        
            Año, por ejemplo: 2010, 2012, etc.

        <em>Retorno</em><br>
        -----------
        Ejemplo:
        ```python
            >>> UsersNotRecommend(2015)
        
        [{"Puesto 1" : ""Counter-Strike: Global Offensive""}, {"Puesto 2" : "DayZ"},{"Puesto 3" : "Rust"}]
        
        ```
        INSTRUCCIONES<br>
                            1. Haga clic en "Try it out".<br>
                            2. Ingrese el Año en el cuadro de abajo. <br>
                            3. Presiona Excute.<br>
                            4. Desplácese hacia "Response body" para ver el resultado.
                            </font>

        """
    # Filtrar el DataFrame por el año dado
    filtered_df = df_top_negative_games_by_year[df_top_negative_games_by_year['reviews_year'] == year]
    
    # Verificar si se encontraron revisiones para el año dado
    if filtered_df.empty:
        return "Año ingresado inválido o sin revisiones."
    
    top_games_list = [
        {"Puesto 1": filtered_df.iloc[0]['Rank 1']},
        {"Puesto 2": filtered_df.iloc[0]['Rank 2']},
        {"Puesto 3": filtered_df.iloc[0]['Rank 3']}
    ]
    return top_games_list


#5
@app.get("/reviews_sentiment_analysis/{year}")
def sentiment_analysis(year: int):
    """
        <font color="blue">
        <h1><u>sentiment_analysis</u></h1>
        </font>

        <b>Análisis de sentimiento, para reviews por año.</b><br>
        <b>Esta función recibe un año y retorna una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.</b><br>

        <em>Parámetros</em><br>
        ----------
        año : <code>int</code>
        
            Año, por ejemplo: 2010, 2012, etc.

        <em>Retorno</em><br>
        -----------
        Ejemplo:
        ```python
            >>> sentiment_analysis(2015)
        
        [{Negative = 928, Neutral = 1908, Positive = 3653}]
        
        ```
        INSTRUCCIONES<br>
                            1. Haga clic en "Try it out".<br>
                            2. Ingrese el Año en el cuadro de abajo. <br>
                            3. Presiona Excute.<br>
                            4. Desplácese hacia "Response body" para ver el resultado.
                            </font>

        """
    # Crear un dataframe donde las filas son años y las columnas análisis de sentimiento.
    df_sentiment_counts = pd.read_csv('./Data_API/df_sentiment_counts.csv')

    # Encontrar los valores de sentimiento para el año pasado como parámetro.
    negativo = df_sentiment_counts.loc[df_sentiment_counts.release_year == year, "total_negativos"].item()
    neutral = df_sentiment_counts.loc[df_sentiment_counts.release_year == year, "total_neutrales"].item()
    positivo = df_sentiment_counts.loc[df_sentiment_counts.release_year == year, "total_positivos"].item()

    return {"Negative" : negativo, "Neutral" : neutral, "Positive" : positivo}


#6
df_item_similarity = pd.read_parquet('./Data_API/df_item_similarity.parquet')
num_recomendaciones=5

@app.get("/top5_recommended_games_name/{item_name}")
def game_recommendation(item_name: str):
    """
        <font color="blue">
        <h1><u>game_recommendation</u></h1>
        </font>

        <b>Lista con 5 juegos recomendados similares al ingresado.</b><br>
        <b>Esta función recibe un nombre de videojuego y retorna una lista con 5 juegos recomendados similares al ingresado.</b><br>

        <em>Parámetros</em><br>
        ----------
        item_name : <code>str</code>
        
            Nombre del videojuego, por ejemplo: Renegade Ops, Pro Evolution Soccer 2013, SimCity 4 Deluxe, etc.

        <em>Retorno</em><br>
        -----------
        Ejemplo:
        ```python
            >>> game_recommendation(2015)
                
                [{
        "Juegos similares a": "Renegade Ops",
        "Recomendaciones": [
            {
            "Recomendación": 1,
            "Juego": "Xpand Rally Xtreme"
            },
            {
            "Recomendación": 2,
            "Juego": "Puzzle Agent"
            },
            {
            "Recomendación": 3,
            "Juego": "Counter-Strike"
            },
            {
            "Recomendación": 4,
            "Juego": "FTL: Faster Than Light"
            },
            {
            "Recomendación": 5,
            "Juego": "Sid Meier's Civilization V"
            }
            ]
            }]
         ```

        INSTRUCCIONES<br>
                            1. Haga clic en "Try it out".<br>
                            2. Ingrese el nombre del videojuego en el cuadro de abajo. (Primera letra Mayuscula, ejemplo: SimCity 4 Deluxe)<br>
                            3. Presiona Excute.<br>
                            4. Desplácese hacia "Response body" para ver el resultado.
                            </font>

        """
    if item_name in df_item_similarity:
        juego_similaridades = df_item_similarity[item_name]
        juegos_similares = juego_similaridades.sort_values(ascending=False)
        juegos_similares = juegos_similares[1:num_recomendaciones + 1]
        recommendations = [{"Recomendación": i, "Juego": juego} for i, (juego, similitud) in enumerate(juegos_similares.items(), 1)]
        return JSONResponse(content={"Juegos similares a": item_name, "Recomendaciones": recommendations})
    else:
        return JSONResponse(content={"Mensaje": "El juego ingresado no está en la base de datos."})
