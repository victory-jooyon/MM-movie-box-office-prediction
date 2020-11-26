import torch
import torch.nn as nn
from .feature_extractors.imdb import IMDBFeatureExtractor
from .feature_extractors.tmdb import TMDBFeatureExtractor
from .feature_extractors.poster import PosterFeatureExtractor


class MultimodalPredictionModel(nn.Module):
    def __init__(self, feature_size=768, hidden_layer_size=512):
        super(MultimodalPredictionModel, self).__init__()
        self.tmdb = TMDBFeatureExtractor(feature_size)
        self.poster = PosterFeatureExtractor(feature_size)
        self.imdb = IMDBFeatureExtractor(feature_size)

        self.fc = nn.Sequential(nn.Linear(3 * feature_size, hidden_layer_size),
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(hidden_layer_size, 1))

    def forward(self, tmdb_tok, poster_input, imdb_tok):
        # [feature_size]
        tmdb_feature = self.tmdb(tmdb_tok)
        poster_feature = self.poster(poster_input)
        imdb_feature = self.imdb(imdb_tok)

        # [feature_size * 3]
        total_feature = torch.cat((tmdb_feature, poster_feature, imdb_feature), dim=1)

        # [1]
        output = self.fc(total_feature)

        return output
