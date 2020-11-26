import json
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from transformers import BertTokenizer


class MovieDataset(Dataset):
    def __init__(self, mode='train'):
        super(Dataset, self).__init__()
        # Read data from json
        with open('./data/all_data.json', 'r', encoding='utf-8') as f:
            movie_data = json.load(f)
        self.ids, self.poster_urls, self.reviews, self.overviews, self.revenues = [], [], [], [], []
        for data in tqdm(movie_data, total=len(movie_data), desc=f'Loading {mode} dataset'):
            # if not data['tmdb']['reviews']:
            #     continue
            # self.ids.append(data['id'])
            # self.poster_urls.append(data['tmdb']['poster'])
            # self.overviews.append(data['tmdb']['overview'])
            # self.reviews.append(data['tmdb']['reviews'][0])
            # self.revenues.append(int(data['revenue']))
            self.ids.append(data['id'])
            self.poster_urls.append(data['poster'])
            self.overviews.append(data['overview'])
            self.reviews.append(data['review'])
            self.revenues.append(data['revenue'])

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
        self.reviews = self.reviews[split_range]
        self.overviews = self.overviews[split_range]
        self.revenues = self.revenues[split_range]

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
        length = inputs['attention_mask'].sum().item()
        return inputs, length

    def __getitem__(self, idx):
        # Get data
        url, review, overview = self.poster_urls[idx], self.reviews[idx], self.overviews[idx]
        revenue = self.revenues[idx]

        # Preprocess data
        res = requests.get('https://image.tmdb.org/t/p/original' + url, stream=True)
        image = Image.open(BytesIO(res.content))
        image = self.image_process(image)
        review, review_len = self.get_tokenized(review)
        overview, overview_len = self.get_tokenized(overview)
        revenue = torch.tensor(revenue)

        return image, review, overview, revenue
