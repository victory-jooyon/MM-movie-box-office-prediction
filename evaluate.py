from tqdm import tqdm

import torch

class Evaluator:
    def __init__(self, args, model, loader, criterion):
        self.args = args
        self.model = model
        self.loader = loader
        self.criterion = criterion

    def evaluate(self, mode=''):
        self.model.eval()
        pbar = tqdm(self.loader, total=len(self.loader), desc=mode)
        total_loss, n_data = 0, 0
        for i, data in enumerate(pbar):
            # Load data
            poster, overview, true_revenue, imdb = data
            poster, true_revenue = poster.to(self.args.device), true_revenue.to(self.args.device)
            for key in overview.keys():
                overview[key] = overview[key].to(self.args.device)
            for key in imdb.keys():
                imdb[key] = imdb[key].to(self.args.device)
            # Forward model & Get loss
            with torch.no_grad():
                pred_revenue = self.model(overview, poster, imdb)
                loss = self.criterion(pred_revenue, true_revenue)

            total_loss += loss.item()
            n_data += poster.shape[0]

        avg_loss = float(total_loss) / n_data
        print(f'{mode}: Average Loss: {avg_loss}')
        return avg_loss

