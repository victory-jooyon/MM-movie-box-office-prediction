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
        self.best_val_acc = 0
        self.test_acc = 0

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
                    if self.args.aug == 'mlp' or self.args.aug == 'pool-max':
                        imdb = imdb.to(self.args.device)
                    else:
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

            print(f"Average training loss: {(total_loss / total_data):.6f}")
            # Validate for some interval
            if epoch % self.args.valid_interval == 0:
                valid_loss, valid_acc = self.validation.evaluate(f'Epoch {epoch} validation')

                if self.best_val_acc < valid_acc:
                    _, test_acc = self.test_evaluator.evaluate(f'Epoch {epoch} test')
                    self.best_val_acc = valid_acc
                    self.test_acc = test_acc
                    print(f'Test Acc renewed: {test_acc:.6f}')
                print(f'Epoch {epoch}: Loss: {valid_loss:.6f} | Average Acc: {valid_acc:.6f} | Best Acc: {self.best_val_acc} | corresp. test Acc: {self.test_acc}')


                if self.args.show_example:
                    self.test_evaluator.predict_example()

        print('Train end!')








