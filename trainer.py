import os

import torch

from tqdm import tqdm

from evaluation import evaluate


class Trainer:
    def __init__(self, args, model, train_loader, valid_loader, criterion, optimizer):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.validation = evaluate.Evaluator(args, model, valid_loader, criterion)

    def train(self):
        best_loss = float('inf')
        for epoch in self.args.epochs:
            # Train
            self.model.train()
            pbar = tqdm(self.train_loader, total=len(self.train_loader), desc=f'Epoch {epoch} training')
            for i, data in enumerate(pbar):
                # Load data
                poster, review, overview, true_revenue = data
                poster, review = poster.to(self.args.device), review.to(self.args.device)
                overview, true_revenue = overview.to(self.args.device), true_revenue.to(self.args.device)

                # Forward model & Get loss
                pred_revenue = self.model(poster, review, overview)
                loss = self.criterion(pred_revenue, true_revenue)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validate for some interval
            if epoch % self.args.valid_interval == 0:
                valid_loss = self.validation.evaluate(f'Epoch {epoch} validation')
                if valid_loss < best_loss:
                    torch.save(self.model.state_dict(), os.path.join(self.args.weight_dir, 'best_checkpoint.pt'))

        torch.save(self.model.state_dict(), os.path.join(self.args.weight_dir, 'last_checkpoint.pt'))
        print('Train end!')








