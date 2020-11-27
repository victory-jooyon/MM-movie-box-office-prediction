import json

import requests
from tqdm import tqdm


filtered = []

with open('data.json') as f:
    data = json.load(f)
    for d in tqdm(data):
        if d['revenue'] == 0:
            continue

        url = d['tmdb']['poster']
        res = requests.get(url)
        if res.status_code == 404:
            continue

        try:
            d['imdb']['release_year'] = d['imdb']['release_date'].split('-')[0]
            d['imdb']['main_actor'] = d['imdb']['actors'].split(', ')[0]
            filtered.append(d)
        except Exception as ex:  # 에러 종류
            print(f"Error on {d['id']}:", ex)

with open('../validated_data.json', 'w', encoding='utf-8') as f:
    json.dump(filtered, f, indent=4)

