import torch
import torch.nn as nn
from .feature_extractors.tmdb import TMDBFeatureExtractor
from .feature_extractors.poster import PosterFeatureExtractor
from .feature_extractors.imdb import IMDBFeatureExtractor


class MultimodalPredictionModel(nn.Module):
    def __init__(self, feature_size=768, hidden_layer_size=512, ablation=None):
        super(MultimodalPredictionModel, self).__init__()
        self.tmdb = TMDBFeatureExtractor(feature_size)
        self.poster = PosterFeatureExtractor(feature_size)
        self.imdb = IMDBFeatureExtractor(feature_size)
        self.ablation = ablation

        fc_size = feature_size * 2 if ablation is None else feature_size
        self.fc = nn.Sequential(nn.Linear(fc_size, hidden_layer_size),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(hidden_layer_size, 1))

    def forward(self, tmdb_tok, poster_input, imdb_tok):
        # [feature_size]
        tmdb_feature = self.tmdb(tmdb_tok)
        poster_feature = self.poster(poster_input)
        imdb_feature = self.imdb(imdb_tok)

        # [feature_size * 3]
        if self.ablation is None:
            total_feature = torch.cat((tmdb_feature, poster_feature, imdb_feature), dim=1)
        elif self.ablation == 'poster':
            total_feature = poster_feature
        elif self.ablation == 'tmdb':
            total_feature = tmdb_feature
        else: # imdb
            total_feature = imdb_feature

        # [1]
        output = self.fc(total_feature)

        return output
