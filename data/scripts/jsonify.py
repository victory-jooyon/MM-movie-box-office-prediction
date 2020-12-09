from __future__ import print_function, division
import csv
import os
import json

from tqdm import tqdm


class Movie:
    def __init__(self, id, tmdb_id):
        self.id = id
        self.imdb = {
            'imdb_id': None,
        }
        self.tmdb = {
            'tmdb_id': tmdb_id,
        }
        self.revenue = None


def get_csv_reader(filename, delimiter):
    reader = []
    if not os.path.isfile(filename):
        csvfile = open(filename, "w")
    else:
        csvfile = open(filename, "rt")
        reader = csv.DictReader(csvfile, delimiter=delimiter)
    return list(reader)


all_movies = []

links = get_csv_reader('links.csv', ',')
movies_metadata = get_csv_reader('movies_metadata.csv', ',')
imdb_movies = get_csv_reader('imdb_movies.csv', ',')
imdb_ratings = get_csv_reader('imdb_ratings.csv', ',')


for link in tqdm(links):
    movie = Movie(link['movieId'], link['tmdbId'])
    all_movies.append(movie)


for movie_metadata in tqdm(movies_metadata):
    id = movie_metadata['id']
    for movie in all_movies:
        if id == movie.id:
            movie.imdb['imdb_id'] = movie_metadata['imdb_id']
            movie.imdb['title'] = movie_metadata['original_title']
            movie.imdb['release_date'] = movie_metadata['release_date']
            movie.imdb['runtime'] = movie_metadata['runtime']

            movie.revenue = movie_metadata['revenue']

            movie.tmdb['overview'] = movie_metadata['overview']
            movie.tmdb['budget'] = movie_metadata['budget']
            movie.tmdb['genres'] = movie_metadata['genres']
            movie.imdb['language'] = movie_metadata['original_language']
            movie.imdb['popularity'] = movie_metadata['popularity']
            movie.tmdb['poster'] = 'https://image.tmdb.org/t/p/original' + movie_metadata['poster_path']
            movie.tmdb['production_companies'] = movie_metadata['production_companies']
            movie.tmdb['production_countries'] = movie_metadata['production_countries']
            movie.tmdb['vote_average'] = movie_metadata['vote_average']
            movie.tmdb['vote_count'] = movie_metadata['vote_count']

            break


for imdb_movie in tqdm(imdb_movies):
    imdb_id = imdb_movie['imdb_title_id']
    for movie in all_movies:
        if imdb_id == movie.imdb['imdb_id']:
            movie.imdb['genre'] = imdb_movie['genre']
            movie.imdb['director'] = imdb_movie['director']
            movie.imdb['writer'] = imdb_movie['writer']
            movie.imdb['actors'] = imdb_movie['actors']
            movie.imdb['description'] = imdb_movie['description']

            break

for imdb_rating in tqdm(imdb_ratings):
    imdb_id = imdb_movie['imdb_title_id']
    for movie in all_movies:
        if imdb_id == movie.imdb['imdb_id']:
            movie.imdb['weighted_average_vote'] = imdb_rating['weighted_average_vote']
            movie.imdb['total_votes'] = imdb_rating['total_votes']
            movie.imdb['us_voters_rating'] = imdb_rating['us_voters_rating']
            movie.imdb['non_us_voters_rating'] = imdb_rating['non_us_voters_rating']

            break


data = []
for movie in tqdm(all_movies):
    if movie.revenue is not None and movie.imdb['imdb_id'] is not None:
        data.append({
            'id': movie.id,
            'imdb': movie.imdb,
            'tmdb': movie.tmdb,
            'revenue': movie.revenue,
        })


with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)
