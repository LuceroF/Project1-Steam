# Steam Video Game Recommendation Project

## Overview

At the heart of Steam, this project emerges to enhance the gaming experience with a **customized recommendation system**, designed to adapt and evolve according to our users' preferences. As a Data Scientist, I've blended **Machine Learning** techniques not just to refine recommendations but also to distill deep analysis of gaming behavior. These analyses are crucial, providing a rich database for strategic product development and updates. The goal is to keep Steam as an innovation leader, ensuring that every gamer easily finds their next favorite game and that each title gets a chance to shine.

## Technologies and Tools Used

- **Programming Language:** Python
- **Main Libraries and Frameworks:**
  - FastAPI for API development
  - NLTK for sentiment analysis
  - _Other libraries include matplotlib, pandas, numpy, seaborn, etc._

## Development Process

### ETL

The ETL process was carried out using three main databases in gzip format: `user_items`, `user_reviews`, and `steam_games`. Data was cleaned and transformed, then stored in parquet format to optimize performance in the API and ML model. (See the attached file: [ETL.ipynb](/ETL.ipynb) for more details).

### ML Model Training and Maintenance

The recommendation model is based on an item-item approach. Relevant features were selected, and cosine similarity was used to generate recommendations for similar games. (See the attached file: [Functions.ipynb](/Functions.ipynb) for more details).

### ML Model Features

The `item-item` recommendation system is implemented through the endpoint `/game_recommendation/{item_id}`, which returns a list of 5 recommended games similar to the input game based on its ID.

## API and Endpoints

The API, developed with FastAPI, exposes several endpoints for accessing the information and functionalities of the system:

1. **PlayTimeGenre**
2. **UserForGenre**
3. **UsersRecommend**
4. **UsersWorstDeveloper**
5. **sentiment_analysis**

The implementation of the API and endpoints is detailed in the main.py file. (See [main.py](/main.py) for more information).

## Deployment

The deployment was carried out on [Render.com](https://render.com/), connecting the GitHub repository with the platform for continuous and efficient deployment. You can see it here: [steam-mlops-utyg.onrender.com](https://steam-mlops-utyg.onrender.com/docs).

## Exploratory Data Analysis (EDA)

The EDA provided valuable insights into the relationships between variables, patterns, and possible anomalies in the data. (See the attached file: [EDA.ipynb](/EDA.ipynb) for more details).

## Project Highlights

- The project's foundation is a **dynamic RESTful API**, designed to be universally accessible.
- Implementation was carried out using **FastAPI**, a strategic choice to ensure agile and stable API development.
- Deployment was performed on **Render**, selected for its efficiency and simplicity in continuous integration.
- The **exploratory data analysis** phase was approached manually, providing a detailed understanding and deep knowledge of our datasets.

## Final Reflections and Future Perspectives

In this project, it has been demonstrated how **Data Engineering** and **Machine Learning** disciplines can be applied to solve real-world problems in the business realm of video games on **Steam**. Every step and resource used has been documented, facilitating replication and a comprehensive understanding of the proposed approach. The entire process and results are available in the **detailed repository** that has been created.

The project has also allowed exploring the possibilities of optimization and advancement with additional resources on **Render**. An efficient and agile system has been developed, deployed on **Render.com** with an intuitive interface and complete documentation. These actions have enhanced the user experience and provided valuable lessons on the practical implementation of engineering and machine learning systems in real production environments.

This project has been both a challenge and an opportunity to grow professionally and contribute to the realm of video games on **Steam**.
