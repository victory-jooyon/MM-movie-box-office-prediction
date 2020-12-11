import torch
import torch.nn as nn
from transformers import BertModel


class IMDBFeatureExtractor(nn.Module):
    def __init__(self, feature_size, augmode):
        super(IMDBFeatureExtractor, self).__init__()
        self.augmode = augmode
        #if augmode == 'mlp' or augmode == 'pool-max':
        #    self.feature_extractor = nn.Sequential(
        #        nn.Linear(4, 512),
        #        nn.ReLU(),
        #        nn.Dropout(),
        #        nn.Linear(512, 768)
        #    )
        if augmode == 'mlp':
            self.feature_extractor = nn.Sequential(
                nn.Linear(4, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 768)
            )
        else:
            self.feature_extractor = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        for param in self.feature_extractor.parameters():
            if augmode == 'allow-grad' or augmode == 'mlp' or augmode == 'pool-max':
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        if self.augmode == 'mlp' or self.augmode == 'pool-max':
            return self.feature_extractor(x)
        else:
            outputs = self.feature_extractor(**x)

            output = outputs[1]
            return output