import os
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

import evaluate


class Trainer:
    def __init__(self, args, model, train_loader, valid_loader, criterion, optimizer):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.validation = evaluate.Evaluator(args, model, valid_loader, criterion)
        self.writer = SummaryWriter(log_dir='./logs')

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

            self.writer.add_scalar('Loss/train', float(total_loss) / total_data, epoch)

            # Validate for some interval
            if epoch % self.args.valid_interval == 0:
                valid_loss, valid_acc = self.validation.evaluate(f'Epoch {epoch} validation')
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.args.weight_dir, 'best_checkpoint_loss.pt'))
                    print(f'Best Loss Checkpoint Saved at {epoch}')
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    torch.save(self.model.state_dict(), os.path.join(self.args.weight_dir, 'best_checkpoint_acc.pt'))
                    print(f'Best Acc Checkpoint Saved at {epoch}')
                self.writer.add_scalar('Loss/valid', valid_loss, epoch)
                self.writer.add_scalar('Acc/valid', valid_acc, epoch)

        torch.save(self.model.state_dict(), os.path.join(self.args.weight_dir, 'last_checkpoint.pt'))
        self.writer.close()
        print('Train end!')








