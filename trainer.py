import os
from tqdm import tqdm

import torch

import evaluate


class Trainer:
    def __init__(self, args, model, train_loader, valid_loader, criterion, optimizer, test_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.validation = evaluate.Evaluator(args, model, valid_loader, criterion)
        self.test_evaluator = evaluate.Evaluator(args, model, test_loader, criterion)

    def train(self):
        best_loss, best_acc = float('inf'), 0
        for epoch in range(self.args.epochs):
            # Train
            self.model.train()
            pbar = tqdm(self.train_loader, total=len(self.train_loader), desc=f'Epoch {epoch} training')
            total_loss, total_data = 0, 0
            for i, data in enumerate(pbar):
                # Load data
                poster, overview, true_success, imdb = data
                poster, true_success = poster.to(self.args.device), true_success.to(self.args.device)
                for key in overview.keys():
                    overview[key] = overview[key].to(self.args.device)
                    imdb[key] = imdb[key].to(self.args.device)

                # Forward model & Get loss
                pred_success = self.model(overview, poster, imdb)
                loss = self.criterion(pred_success, true_success)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_data += true_success.shape[0]

            # Validate for some interval
            if epoch % self.args.valid_interval == 0:
                valid_loss, valid_acc = self.validation.evaluate(f'Epoch {epoch} validation')
                self.test_evaluator.evaluate(f'Epoch {epoch} test')
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print(f'Best Loss Checkpoint Saved at {epoch}')
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    print(f'Best Acc Checkpoint Saved at {epoch}')

        print('Train end!')








