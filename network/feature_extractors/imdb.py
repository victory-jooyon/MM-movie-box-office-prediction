import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class IMDBFeatureExtractor(nn.Module):
    def __init__(self, feature_size):
        super(IMDBFeatureExtractor, self).__init__()
        self.feature_extractor = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(768, feature_size)

    def forward(self, x):
        outputs = self.feature_extractor(**x)

        # [13 x num_sentence x num_word x 768]
        hidden_states = outputs[2]

        # [num_sentence x num_word x 768]
        sentence_vecs = hidden_states[-2]

        # [num_word x 768]
        word_embeddings = torch.mean(sentence_vecs, dim=0)

        # [768]
        sentence_embedding = torch.mean(word_embeddings, dim=0)

        # [feature_size]
        output = self.fc(sentence_embedding)
        return output
