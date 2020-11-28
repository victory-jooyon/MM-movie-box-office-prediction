import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

import torch
import torchvision.transforms as transforms

from transformers import BertTokenizer
from data.dataset import get_stats

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

    def predict_example(self):
        poster_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def get_tokenized(text, device):
            inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True)
            for key in inputs:
                inputs[key] = torch.tensor(inputs[key]).unsqueeze(0).to(device)
            return inputs

        revenue_mean, revenue_std = get_stats()

        self.model.eval()
        for example in self.example:
            # Preprocess poster
            res_poster = requests.get(example['poster'])
            poster = Image.open(BytesIO(res_poster.content)).convert('RGB')
            poster = poster_transform(poster).to(self.args.device).unsqueeze(0)

            # Preprocess overview
            overview = get_tokenized(example['overview'], device=self.args.device)

            # Preprocess IMDB
            imdb = f"year is {example['release_year']}, genre is {example['genre']}, " \
                   f"director is {example['director']}, actor is {example['main_actor']}"
            imdb = get_tokenized(imdb, device=self.args.device)

            # Preprocess revenue
            revenue = torch.tensor([example['revenue']], dtype=torch.float32, device=self.args.device).unsqueeze(0)
            revenue = (revenue - revenue_mean) / revenue_std

            with torch.no_grad():
                pred_revenue = self.model(overview, poster, imdb)
                loss = self.criterion(pred_revenue, revenue)

            print(f'Evaluate with example {example["title"]}:\n'
                  f'Loss: {loss.item()}\n'
                  f'Predicted Revenue: {(pred_revenue.item() * revenue_std) + revenue_mean}\n'
                  f'Real Revenue: {example["revenue"]}\n\n')
        print('Evaluation Finished')

    example = [
        {
            'id': 299534,
            'title': 'Avengers: Endgame',
            'poster': 'https://image.tmdb.org/t/p/w1280/z7ilT5rNN9kDo8JZmgyhM6ej2xv.jpg',
            'overview': "After the devastating events of Avengers: Infinity War, the universe is in ruins due to the "
                        "efforts of the Mad Titan, Thanos. With the help of remaining allies, the Avengers must "
                        "assemble once more in order to undo Thanos' actions and restore order to the universe once "
                        "and for all, no matter what consequences may be in store.",
            'genre': 'Adventure, Science Fiction, Action',
            'release_year': 2019,
            'main_actor': 'Robert Downey Jr.',
            'director': 'Anthony Russo, Joe Russo',
            'revenue': 2797800564,
        },
        {
            'id': 150540,
            'title': 'Inside Out',
            'poster': 'https://image.tmdb.org/t/p/w1280/qVdrpBY920kKikUmPm89wig60Kd.jpg',
            'overview': "Growing up can be a bumpy road, and it's no exception for Riley, who is uprooted from her "
                        "Midwest life when her father starts a new job in San Francisco. Riley's guiding emotions— Joy, "
                        "Fear, Anger, Disgust and Sadness—live in Headquarters, the control centre inside Riley's mind, "
                        "where they help advise her through everyday life and tries to keep things positive, "
                        "but the emotions conflict on how best to navigate a new city, house and school.",
            "genre": "Animation, Adventure, Comedy",
            "director": "Pete Docter, Ronnie Del Carmen",
            "main_actor": "Amy Poehler",
            'release_year': 2015,
            'revenue': 857611174,
        },
        {
            'id': 109445,
            'title': 'Frozen',
            'poster': 'https://image.tmdb.org/t/p/w1280/nelAGS4rcZm2Qyuy3TSNWgU2mEL.jpg',
            'overview': "Young princess Anna of Arendelle dreams about finding true love at her sister Elsa’s coronation. "
                        "Fate takes her on a dangerous journey in an attempt to end the eternal winter that has fallen "
                        "over the kingdom. She's accompanied by ice delivery man Kristoff, his reindeer Sven, and snowman "
                        "Olaf. On an adventure where she will find out what friendship, courage, family, and true love "
                        "really means.",
            'genre': 'Animation, Adventure, Family',
            'release_year': 2013,
            'director': 'Chris Buck, Jennifer Lee',
            'main_actor': 'Chris Bell',
            'revenue': 1274219009,
        }
    ]
