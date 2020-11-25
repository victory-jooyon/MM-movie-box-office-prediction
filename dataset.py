import torch
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, mode='train'):
        super(Dataset, self).__init__()
        poster_urls, reviews, overviews, revenues = [], [], [], []
        self.urls = poster_urls
        self.reviews = reviews
        self.overviews = overviews
        self.revenues = revenues

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url, review, overview = self.urls[idx], self.reviews[idx], self.overviews[idx]
        revenue = self.revenues[idx]
        return url, review, overview, revenue
