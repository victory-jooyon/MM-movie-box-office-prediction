import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from network.network import MultimodalPredictionModel
from dataset import MovieDataset
from trainer import Trainer
from transformers import set_seed

try:
    import nsml
    USE_NSML = True
except:
    USE_NSML = False


def main():
    t0 = time.time()

    # Arguments
    args = parse_args()

    # Load dataset
    train_loader = DataLoader(MovieDataset(args, split='train', seed=args.seed, num_classes=args.num_classes), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(MovieDataset(args, split='valid', seed=args.seed, num_classes=args.num_classes), batch_size=32,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(MovieDataset(args, split='test', seed=args.seed, num_classes=args.num_classes), batch_size=32,
                             shuffle=False, num_workers=args.num_workers)

    # Load model
    model = MultimodalPredictionModel(args=args, ablation=args.ablation, num_classes=args.num_classes).to(args.device)
    model = nn.DataParallel(model)

    # Load criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1E-4)

    # Run train
    trainer = Trainer(args, model, train_loader, valid_loader, criterion, optimizer, test_loader)
    trainer.train()

    print(f'Entire pipeline Finished\n'
          f'Time elapsed: {time.time() - t0:.4f}')


def parse_args():
    parser = argparse.ArgumentParser('Multimodal box office prediction')
    parser.add_argument('--epochs', default=10, type=int, help='Total epochs to train')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--valid_interval', default=1, type=int, help='Validation interval')
    parser.add_argument('--lr', default=5e-04, type=float, help='Train learning rate')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for loader')
    parser.add_argument('--ablation', default=None, type=str, choices=['poster', 'tmdb', 'imdb', None],
                        help='Where to use single feature for prediction')
    parser.add_argument('--seed', default=1, type=int, help='Dataset shuffle & model seed')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--show_example', action='store_true', help='Whether to show example')

    parser.add_argument('--aug', type=str, choices=['pool-vec', 'normal', 'allow-grad', 'more-layer', 'mlp'])
    args, _ = parser.parse_known_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.USE_NSML = USE_NSML
    set_seed(args.seed)
    print(args)

    return args


if __name__ == '__main__':
    main()
