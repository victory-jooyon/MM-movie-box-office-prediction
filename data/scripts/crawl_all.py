import argparse
import csv
import json
import os
import pathlib

import requests
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Crawling')

    TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
    RAW_DIR = pathlib.Path().absolute().parent

    parser.add_argument('--start', default=0, type=int, help='IMDB starting point (total 4175405 data)')
    parser.add_argument('--end', default=4175405, type=int, help='IMDB starting point (total 4175405 data)')
    parser.add_argument('--jsonfile', default=f'{RAW_DIR}/json/imdb_data.json', type=str, help='IMDB starting point (total 4175405 data)')

    args, _ = parser.parse_known_args()

    with open(args.jsonfile, 'r', encoding='utf-8') as f:
        filtered_movies = json.load(f)

    keys = filtered_movies.keys()
    keys = list(keys)[args.start:args.end+1]

    all_data = []
    last_update = 0
    for i, k in enumerate(tqdm(keys)):

        if i // 10000 == (last_update + 1):
            print('all_data len:', len(all_data))
            filename = f'{RAW_DIR}/json/crawled_data/crawled_data_{args.start}-{args.start + i}_{len(all_data)}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=4)
            last_update += 1
            print(f"Saved at {filename}")

        tmdb_url = f"https://api.themoviedb.org/3/find/{k}?api_key={TMDB_API_KEY}&language=en-US&external_source=imdb_id"
        tmdb_res = json.loads(requests.get(tmdb_url).text)
        rs = tmdb_res['movie_results']
        if rs:
            r = rs[0]
        else:
            r = None
            continue

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
            print(k, r, 'error:', e)

    print('all_data len:', len(all_data))
    filename = f'{RAW_DIR}/json/crawled_data/crawled_data_{args.start}-{args.start + i}_{len(all_data)}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4)
    last_update += 1
    print(f"Saved at {filename}")
