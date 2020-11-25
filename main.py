import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .model.model import SomeModel
from .dataset import MovieDataset
from .trainer import Trainer
from .evaluation.evaluate import Evaluator


def main():
    t0 = time.time()

    # Arguments
    args = parse_args()

    # Load dataset
    train_loader = DataLoader(MovieDataset(mode='train'), args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(MovieDataset(mode='valid'), args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(MovieDataset(mode='test'), args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Load model
    model = SomeModel().to(args.device)
    model = nn.DataParallel(model)

    # Load criterion & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1E-4)

    # Run train
    trainer = Trainer(args, model, train_loader, valid_loader, criterion, optimizer)
    trainer.train()

    # Run test
    evaluator = Evaluator(args, model, test_loader, criterion)
    test_loss = evaluator.evaluate('Test')
    print(f'Entire pipeline Finished\n'
          f'Time elapsed: {time.time() - t0:.4f}\n'
          f'Test Loss: {test_loss}')


def parse_args():
    parser = argparse.ArgumentParser('Multimodal box office prediction')
    parser.add_argument('--epochs', default=30, type=int, help='Total epochs to train')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--valid_interval', default=1, type=int, help='Validation interval')
    parser.add_argument('--lr', default=0.0001, type=float, help='Train learning rate')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for loader')
    parser.add_argument('--device', default='cuda:0', type=str, help='Training device: cpu/cuda/cuda:0,1,...')

    parser.add_argument('--weight_dir', default='./weight', type=str, help='Weight save/load directory')
    parser.add_argument('--resume', default=None, type=str, choices=['best', 'last', None],
                        help='Where to resume')
    parser.add_argument('--data_dir', default='./data', type=str, help='Data save/load directory')

    args, _ = parser.parse_known_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    return args


if __name__ == '__main__':
    main()
