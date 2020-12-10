import os
import shutil
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from network.network import MultimodalPredictionModel
from dataset import MovieDataset
from trainer import Trainer
from evaluate import Evaluator


def main():
    t0 = time.time()

    # Arguments
    args = parse_args()

    # Load dataset
    train_loader = DataLoader(MovieDataset(split='train', seed=args.seed), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(MovieDataset(split='valid', seed=args.seed), batch_size=32,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(MovieDataset(split='test', seed=args.seed), batch_size=32,
                             shuffle=False, num_workers=args.num_workers)

    # Load model
    model = MultimodalPredictionModel(ablation=args.ablation).to(args.device)
    model = nn.DataParallel(model)

    if args.resume is not None:
        model.load_state_dict(torch.load(os.path.join(args.weight_dir, args.resume + '.pt'), map_location=args.device))

    # Load criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1E-4)

    # Run train
    shutil.rmtree('./logs')
    trainer = Trainer(args, model, train_loader, valid_loader, criterion, optimizer)
    trainer.train()

    # Run test
    model.load_state_dict(torch.load(os.path.join(args.weight_dir, 'best_checkpoint_loss.pt'), map_location=args.device))
    evaluator = Evaluator(args, model, test_loader, criterion)
    evaluator.evaluate('Test-BestValLoss')

    model.load_state_dict(torch.load(os.path.join(args.weight_dir, 'best_checkpoint_acc.pt'), map_location=args.device))
    evaluator = Evaluator(args, model, test_loader, criterion)
    evaluator.evaluate('Test-BestValAcc')

    model.load_state_dict(torch.load(os.path.join(args.weight_dir, 'last_checkpoint.pt'), map_location=args.device))
    evaluator = Evaluator(args, model, test_loader, criterion)
    evaluator.evaluate('Test-BestValAcc')

    if args.show_example:
        evaluator.predict_example()

    print(f'Entire pipeline Finished\n'
          f'Time elapsed: {time.time() - t0:.4f}')


def parse_args():
    parser = argparse.ArgumentParser('Multimodal box office prediction')
    parser.add_argument('--epochs', default=10, type=int, help='Total epochs to train')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--valid_interval', default=1, type=int, help='Validation interval')
    parser.add_argument('--lr', default=0.0001, type=float, help='Train learning rate')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for loader')
    parser.add_argument('--ablation', default=None, type=str, choices=['poster', 'tmdb', 'imdb', None],
                        help='Where to use single feature for prediction')
    parser.add_argument('--device', default='cuda:0', type=str, help='Training device: cpu/cuda/cuda:0,1,...')
    parser.add_argument('--seed', default=1, type=int, help='Dataset shuffle seed')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes')

    parser.add_argument('--weight_dir', default='./weight', type=str, help='Weight save/load directory')
    parser.add_argument('--resume', default=None, type=str, choices=['best', 'last', None],
                        help='Where to resume')
    parser.add_argument('--show_example', action='store_true', help='Whether to show example')

    args, _ = parser.parse_known_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.ablation is not None:
        args.weight_dir = os.path.join(args.weight_dir, args.ablation)
    else:
        args.weight_dir = os.path.join(args.weight_dir, 'main')
    os.makedirs(args.weight_dir, exist_ok=True)

    return args


if __name__ == '__main__':
    main()
