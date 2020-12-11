import torch
import torch.nn as nn
from .feature_extractors.tmdb import TMDBFeatureExtractor
from .feature_extractors.poster import PosterFeatureExtractor
from .feature_extractors.imdb import IMDBFeatureExtractor


class MultimodalPredictionModel(nn.Module):
    def __init__(self, args, feature_size=768, hidden_layer_size=512, ablation=None, num_classes=2):
        super(MultimodalPredictionModel, self).__init__()
        self.args = args
        self.tmdb = TMDBFeatureExtractor(feature_size, args.aug)
        self.poster = PosterFeatureExtractor(feature_size, args.aug)
        self.imdb = IMDBFeatureExtractor(feature_size, args.aug)
        self.ablation = ablation

        fc_size = feature_size * 3 if ablation is None else feature_size
        if self.ablation is None and self.args.aug == 'pool-vec':
            fc_size = feature_size

        if self.args.aug == 'more-layer':
            self.fc = nn.Sequential(nn.Linear(fc_size, hidden_layer_size),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(hidden_layer_size, hidden_layer_size),
                                     nn.ReLU(),
                                     nn.Dropout(),
                                     nn.Linear(hidden_layer_size, num_classes))
        else:
            self.fc = nn.Sequential(nn.Linear(fc_size, hidden_layer_size),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(hidden_layer_size, num_classes))
        if self.args.aug == 'pool-vec':
            self.pool = nn.AvgPool1d(3)
        elif self.args.aug == 'pool-max' or self.args.aug == 'max-only':
            self.pool = nn.MaxPool1d(3)

    def forward(self, tmdb_tok, poster_input, imdb_tok):
        # [feature_size]
        tmdb_feature = self.tmdb(tmdb_tok)
        poster_feature = self.poster(poster_input)
        imdb_feature = self.imdb(imdb_tok)

        # [feature_size * 3]
        if self.ablation is None:
            total_feature = torch.cat((tmdb_feature, poster_feature, imdb_feature), dim=1)
            if self.args.aug == 'pool-vec' or self.args.aug == 'pool-max' or self.args.aug == 'max-only':
                total_feature = self.pool(total_feature.unsqueeze(0)).squeeze(0)
        elif self.ablation == 'poster':
            total_feature = poster_feature
        elif self.ablation == 'tmdb':
            total_feature = tmdb_feature
        else: # imdb
            total_feature = imdb_feature

        # [1]
        output = self.fc(total_feature)

        return output
