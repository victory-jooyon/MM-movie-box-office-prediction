import torch
import torch.nn as nn
from model.feature_extractors import TMDBFeatureExtractor, PosterFeatureExtractor, IMDBFeatureExtractor

class MultimodalPredictionModel(nn.Module):
    def __init__(self, feature_size, hidden_layer_size):
        super(MultimodalPredictionModel, self).__init__()
        self.tmdb = TMDBFeatureExtractor(feature_size)
        self.poster = PosterFeatureExtractor(feature_size)
        self.imdb = IMDBFeatureExtractor(feature_size)

        self.fc = nn.Sequential(nn.Linear(3 * feature_size, hidden_layer_size),
                                nn.ReLU(),
                                nn.dropout(),
                                nn.Linear(hidden_layer_size, 1))

    def forward(self, tmdb_tok, tmdb_seg, poster_input, imdb_tok, imdb_seg):
        #[feature_size]
        tmdb_feature = self.tmdb(tmdb_tok, tmdb_seg)
        poster_feature = self.poster(poster_input)
        tmdb_feature = self.tmdb(imdb_tok, imdb_seg)

        #[feature_size * 3]
        total_feature = torch.cat((tmdb_feature, poster_feature, tmdb_feature), dim=0)

         #[1]
        output = self.fc(total_feature)

        return output
