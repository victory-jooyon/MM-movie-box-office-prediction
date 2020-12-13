import csv
import json
import pathlib

from tqdm import tqdm


"""
The tsv files used below are from https://www.imdb.com/interfaces/ .
To run this code, you should download and unzip the .gz files from the above link,
and put them below MM-movie-box-office-prediction/data/csv/ .
The above directory should be made by your own.
"""

if __name__ == '__main__':
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
    with open(f'{RAW_DIR}/json/imdb_data.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_movies, f, indent=4)
