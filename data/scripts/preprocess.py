import json

from tqdm import tqdm


preprocessed = []
genre_dict = {None: 0}
director_dict = {None: 0}
main_actor_dict = {None: 0}


with open('validated_data.json') as f:
    data = json.load(f)
    for d in tqdm(data):

        # Filter revenue = 0
        if d['revenue'] == 0:
            continue

        # Genre
        genre = d['imdb'].get('genre')
        if genre:
            main_genre = genre.split(', ')[0]

        else:
            main_genre = None

        genre_num = genre_dict.get(main_genre)
        if genre_num is None:
            genre_num = len(genre_dict)
            genre_dict[main_genre] = genre_num

        d['imdb']['genre_num'] = genre_num

        # Director
        director = d['imdb'].get('director')
        director_num = director_dict.get(director)
        if director_num is None:
            director_num = len(director_dict)
            director_dict[director] = director_num

        d['imdb']['director_num'] = director_num

        # Actor
        actor = d['imdb'].get('main_actor')
        actor_num = main_actor_dict.get(actor)
        if actor_num is None:
            actor_num = len(main_actor_dict)
            main_actor_dict[actor] = actor_num

        d['imdb']['main_actor_num'] = actor_num

        preprocessed.append(d)


# with open('preprocessed_data.json', 'w', encoding='utf-8') as f:
#     json.dump(preprocessed, f, indent=4)
