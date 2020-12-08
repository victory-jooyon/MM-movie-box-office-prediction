import json
import random
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from transformers import BertTokenizer


class MovieDataset(Dataset):
    def __init__(self, mode='train', thres=(0, 1), seed=0):
        super(Dataset, self).__init__()
        # Read data from json
        with open('./data/validated_data.json', 'r', encoding='utf-8') as f:
            movie_data = json.load(f)
        self.data_low, self.data_mid, self.data_high = [], [], []
        random.seed(seed)

        for data in tqdm(movie_data, total=len(movie_data), desc=f'Loading {mode} dataset'):
            if data['revenue'] == '0' or data['tmdb']['budget'] == '0':
                continue

            id = int(data['id'])
            poster = data['tmdb']['poster']
            overview = data['tmdb']['overview']

            # Build imdb text
            imdb = []
            for key in ['release_year', 'genre_num', 'director_num', 'main_actor_num']:
                if key in data['imdb']:
                    imdb.append(data['imdb'][key])
                else:
                    imdb.append(' ')
            imdb_text = "year is {}, genre is {}, director is {}, actor is {}".format(*imdb)

            # Label profit
            revenue = float(data['revenue'])
            budget = float(data['tmdb']['budget'])
            profit = (revenue - budget) / budget

            if profit < thres[0]:
                success = 0
                self.data_low.append((id, poster, overview, imdb_text, profit, success))
            elif profit < thres[1]:
                success = 1
                self.data_mid.append((id, poster, overview, imdb_text, profit, success))
            else:
                success = 2
                self.data_high.append((id, poster, overview, imdb_text, profit, success))

        n_data = min(len(self.data_low), len(self.data_mid), len(self.data_high))
        random.shuffle(self.data_low)
        random.shuffle(self.data_mid)
        random.shuffle(self.data_high)

        self.data = self.data_low[:n_data] + self.data_mid[:n_data] + self.data_high[:n_data]
        random.shuffle(self.data)

        # Split data
        total_len = len(self.data)
        train_limit = int(total_len * 0.8)
        valid_limit = int(total_len * 0.9)
        if mode == 'train':
            split_range = slice(train_limit)
        elif mode == 'valid':
            split_range = slice(train_limit, valid_limit)
        else:
            split_range = slice(valid_limit, total_len)
        self.data = self.data[split_range]

        # Preprocessor
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.image_process = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        print(f'Dataset {mode} loaded, number of data: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def get_tokenized(self, text):
        inputs = self.tokenizer(text, padding='max_length', max_length=256, truncation=True)
        for key in inputs:
            inputs[key] = torch.tensor(inputs[key])
        return inputs

    def __getitem__(self, idx):
        # Get data
        movie_id, url, overview, imdb_text, profit, success = self.data[idx]

        # Preprocess data
        res = requests.get(url, stream=True)
        image = Image.open(BytesIO(res.content)).convert('RGB')
        image = self.image_process(image)

        overview = self.get_tokenized(overview)
        imdb = self.get_tokenized(imdb_text)

        success = torch.tensor(success, dtype=torch.long)

        return image, overview, success, imdb
