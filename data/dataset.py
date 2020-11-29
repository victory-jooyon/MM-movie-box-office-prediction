import json
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from transformers import BertTokenizer
from table_bert import TableBertModel, Table, Column

revenues_mean, revenues_std = 0, 1

class MovieDataset(Dataset):
    def __init__(self, mode='train'):
        super(Dataset, self).__init__()
        # Read data from json
        with open('./data/validated_data.json', 'r', encoding='utf-8') as f:
            movie_data = json.load(f)
        self.ids, self.poster_urls, self.overviews, self.revenues = [], [], [], []
        self.imdb_text = []

        for data in tqdm(movie_data, total=len(movie_data), desc=f'Loading {mode} dataset'):
            if data['revenue'] == '0':
                continue
            self.ids.append(int(data['id']))
            self.poster_urls.append(data['tmdb']['poster'])
            self.overviews.append(data['tmdb']['overview'])
            self.revenues.append(int(data['revenue']))

            # Build imdb text
            imdb = []
            for key in ['release_year', 'genre_num', 'director_num', 'main_actor_num']:
                if key in data['imdb']:
                    imdb.append(data['imdb'][key])
                else:
                    imdb.append(' ')

            budget = data['tmdb']['budget']

            # imdb_text = "year is {}, genre is {}, director is {}, actor is {}".format(*imdb, budget)
            imdb_text = data['revenue']
            self.imdb_text.append(imdb_text)

        global revenues_mean, revenues_std
        revenues_mean = np.array(self.revenues).mean()
        revenues_std = np.array(self.revenues).std()
        self.revenues_mean = revenues_mean
        self.revenues_std = revenues_std

        # Split data
        total_len = len(self.ids)
        train_limit = int(total_len * 0.8)
        valid_limit = int(total_len * 0.9)
        if mode == 'train':
            split_range = slice(train_limit)
        elif mode == 'valid':
            split_range = slice(train_limit, valid_limit)
        else:
            split_range = slice(valid_limit, total_len)

        self.ids = self.ids[split_range]
        self.poster_urls = self.poster_urls[split_range]
        self.overviews = self.overviews[split_range]
        self.revenues = self.revenues[split_range]
        self.imdb_text = self.imdb_text[split_range]

        # Preprocessor
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.image_process = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        print(f'Dataset {mode} loaded, number of data: {len(self.ids)}')

    def __len__(self):
        return len(self.ids)

    def get_tokenized(self, text):
        inputs = self.tokenizer(text, padding='max_length', max_length=256, truncation=True)
        for key in inputs:
            inputs[key] = torch.tensor(inputs[key])
        return inputs

    def __getitem__(self, idx):
        # Get data
        url, overview = self.poster_urls[idx], self.overviews[idx]
        revenue = self.revenues[idx]

        # Preprocess data
        res = requests.get(url, stream=True)
        image = Image.open(BytesIO(res.content)).convert('RGB')
        image = self.image_process(image)

        overview = self.get_tokenized(overview)
        revenue = torch.tensor(revenue, dtype=torch.float32).unsqueeze(-1)
        revenue = (revenue - self.revenues_mean) / self.revenues_std

        # process imdb data
        imdb = self.get_tokenized(self.imdb_text[idx])

        return image, overview, revenue, imdb

def get_stats():
    global revenues_mean, revenues_std
    return revenues_mean, revenues_std