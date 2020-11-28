import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class IMDBFeatureExtractor(nn.Module):
    def __init__(self, feature_size):
        super(IMDBFeatureExtractor, self).__init__()
        self.feature_extractor = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.feature_extractor(**x)

        output = outputs[1]
        return output