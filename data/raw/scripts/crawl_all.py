import csv
import json
import os
import pathlib

import requests
from tqdm import tqdm


TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
RAW_DIR = pathlib.Path().absolute().parent

basic_tsv = open(f'{RAW_DIR}/csv/title.basics.tsv')
basic_reader = csv.DictReader(basic_tsv, delimiter='\t')
all_movies = {}
for row in tqdm(basic_reader):
    imdb_id = row['tconst']
    genres = row['genres']
    if not genres:
        continue
    all_movies[imdb_id] = {
        'title': row['originalTitle'],
        'release_year': row['startYear'],
        'main_genre': genres.split(',')[0],
    }

print('all_movies:', len(all_movies))

crew_tsv = open(f'{RAW_DIR}/csv/title.crew.tsv')
crew_reader = csv.DictReader(crew_tsv, delimiter='\t')
for row in tqdm(crew_reader):
    imdb_id = row['tconst']
    directors = row['directors']
    movie_exists = all_movies.get(imdb_id)
    if movie_exists:
        all_movies[imdb_id]['director'] = directors.split(',')[0]


principal_tsv = open(f'{RAW_DIR}/csv/title.principals.tsv')
principal_reader = csv.DictReader(principal_tsv, delimiter='\t')
for row in tqdm(principal_reader):
    imdb_id = row['tconst']
    movie_exists = all_movies.get(imdb_id)
    if movie_exists and row['category'].startswith('act'):
        actor_exists = movie_exists.get('main_actor') is not None
        if not actor_exists:
            all_movies[imdb_id]['main_actor'] = row['nconst']

keys = all_movies.keys()
filtered_movies = {}
for k in keys:
    if len(all_movies[k]) == 5:
        filtered_movies[k] = all_movies[k]

print('filtered_movies:', len(filtered_movies))

page = 1
all_data = []
while page <= 500:
    print('page_num:', page, 'all_data_len:', len(all_data))
    crawl_url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&language=en-US&include_video=false&page={page}"
    crawl_res = json.loads(requests.get(crawl_url).text)
    results = crawl_res['results']

    for r in results:
        try:
            tmdb_id = r['id']
            main_genre = r['genre_ids'][0] if r['genre_ids'] else None
            overview = r.get('overview')
            if not overview:
                continue
            poster_path = r.get('poster_path')
            if poster_path:
                poster_url = "https://image.tmdb.org/t/p/original" + poster_path
            else:
                continue
            vote_count = r.get('vote_count')
            vote_average = r.get('vote_average')

            all_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
            all_res = json.loads(requests.get(all_url).text)
            production_companies = all_res.get('production_companies')
            production_countries = all_res.get('production_countries')
            genres = all_res.get('genres')
            budget = all_res.get('budget', 0)
            revenue = all_res.get('revenue', 0)
            if budget == 0 or revenue == 0:
                continue

            review_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews?api_key={TMDB_API_KEY}&language=en-US"
            review_res = json.loads(requests.get(review_url).text)
            review = review_res['results']
            if review:
                review = review[0]['content']
            else:
                review = None

            imdb_id = all_res.get('imdb_id')
            if not imdb_id:
                continue

            imdb_data = filtered_movies.get(imdb_id)
            if not imdb_data:
                continue

            title = imdb_data.get('title')

            release_year = imdb_data.get('release_year')
            if not release_year:
                continue

            main_actor = imdb_data.get('main_actor')
            if not main_actor:
                continue

            director = imdb_data.get('director')
            if not director:
                continue

            main_genre = imdb_data.get('main_genre')
            if not main_genre:
                continue

            data = {
                "id": tmdb_id,
                "imdb": {
                    "imdb_id": imdb_id,
                    "title": title,
                    "director": director,
                    "release_year": release_year,
                    "main_actor": main_actor,
                    "main_genre": main_genre,
                },
                "tmdb": {
                    "tmdb_id": tmdb_id,
                    "overview": overview,
                    "budget": budget,
                    "genres": genres,
                    "poster": poster_url,
                    "production_companies": production_companies,
                    "production_countries": production_countries,
                    "vote_average": vote_average,
                    "vote_count": vote_count,
                    "review": review,
                },
                "revenue": revenue,
            }
            all_data.append(data)

        except Exception as e:
            print(r, 'error:', e)

    page += 1


with open('../../json/crawled_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=4)
