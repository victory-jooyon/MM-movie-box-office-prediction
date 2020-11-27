import json

import requests
from tqdm import tqdm


filtered = []

with open('data.json') as f:
    data = json.load(f)
    for d in tqdm(data):
        if d['revenue'] == 0:
            continue

        all_url = f"https://api.themoviedb.org/3/movie/{d['id']}?api_key=e99a46c20a016e5c29a140271f4ee5bc&language=en-US"
        all_res = json.loads(requests.get(all_url).text)
        poster_path = all_res.get('poster_path')
        if poster_path:
            poster_url = "https://image.tmdb.org/t/p/original"+poster_path
        else:
            continue
        d['tmdb']['poster'] = poster_url
        review_url = f"https://api.themoviedb.org/3/movie/{d['id']}/reviews?api_key=e99a46c20a016e5c29a140271f4ee5bc&language=en-US"
        
        poster_res = requests.get(poster_url)
        if poster_res.status_code == 404:
            continue
        
        review_res = json.loads(requests.get(review_url).text)
        review = review_res['results']
        if review:
            review = review[0]['content']
        else:
            review = None

        d['tmdb']['review'] = review

        d['imdb']['release_year'] = d['imdb']['release_date'].split('-')[0]
        actors = d['imdb'].get('actors')
        if actors:
            d['imdb']['main_actor'] = actors.split(', ')[0]
        else:
            d['imdb']['main_actor'] = None
        filtered.append(d)

with open('../validated_data.json', 'w', encoding='utf-8') as f:
    json.dump(filtered, f, indent=4)

