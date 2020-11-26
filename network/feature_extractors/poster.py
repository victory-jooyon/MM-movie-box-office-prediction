import torch.nn as nn
import torchvision


class PosterFeatureExtractor(nn.Module):
    def __init__(self, feature_size):
        super(PosterFeatureExtractor, self).__init__()
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, feature_size)

    def forward(self, x):
        # [768]
        output = self.feature_extractor(x)
        return output
