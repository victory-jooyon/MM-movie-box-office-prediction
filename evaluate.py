import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

import torch
import torchvision.transforms as transforms

from transformers import BertTokenizer

class Evaluator:
    def __init__(self, args, model, loader, criterion):
        self.args = args
        self.model = model
        self.loader = loader
        self.criterion = criterion

    def evaluate(self, mode=''):
        self.model.eval()
        pbar = tqdm(self.loader, total=len(self.loader), desc=mode)
        total_loss, total_acc, n_data = 0, 0, 0
        tp, fp, fn = 0, 0, 0
        for i, data in enumerate(pbar):
            # Load data
            poster, overview, success, imdb = data
            poster, success = poster.to(self.args.device), success.to(self.args.device)
            for key in overview.keys():
                overview[key] = overview[key].to(self.args.device)
                imdb[key] = imdb[key].to(self.args.device)

            # Forward model & Get loss
            with torch.no_grad():
                pred_success = self.model(overview, poster, imdb)

            loss = self.criterion(pred_success, success)
            total_acc += torch.eq(torch.argmax(pred_success, dim=1), success).to(torch.float32).sum().item()
            total_loss += loss.item()
            n_data += poster.shape[0]

            if self.args.num_classes == 2:
                tp_tmp = success[torch.eq(torch.argmax(pred_success, dim=1), success)].sum().item()
                tp += tp_tmp
                fp += torch.argmax(pred_success, dim=1).sum().item() - tp_tmp
                fn += (1 - success)[torch.ne(torch.argmax(pred_success, dim=1), success)].sum().item()

        avg_loss = float(total_loss) / n_data
        avg_acc = total_acc / n_data
        print(f'{mode}: Average Loss: {avg_loss:.6f} | Average Acc: {avg_acc:.6f}')
        if self.args.num_classes == 2:
            precision = float(tp) / (tp + fp)
            recall = float(tp) / (tp + fn)
            f1 = 2. / ((1. / precision) + (1. / recall))
            print(f'Binary Label - Precision: {precision} | Recall: {recall} | F1 Score: {f1}')
        return avg_loss, avg_acc

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

        # revenue_mean, revenue_std = get_stats()

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
                  f'Loss: {loss.item():.4f}\n'
                  f'Predicted Revenue: {int((pred_revenue.item() * revenue_std) + revenue_mean)}\n'
                  f'Real Revenue: {int(example["revenue"])}\n\n')
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
        },
        {
            'id': 546554,
            'title': 'Knives Out',
            'poster': 'https://image.tmdb.org/t/p/w600_and_h900_bestv2/6rthGOcMZgUUaWN4FGSTd3n3ovg.jpg',
            'overview': "When renowned crime novelist Harlan Thrombey is found dead at his estate just after his 85th "
                        "birthday, the inquisitive and debonair Detective Benoit Blanc is mysteriously enlisted to "
                        "investigate. From Harlan's dysfunctional family to his devoted staff, Blanc sifts through a "
                        "web of red herrings and self-serving lies to uncover the truth behind Harlan's untimely "
                        "death.",
            'genre': 'Mystery, Comedy, Drama, Crime',
            'release_year': 2019,
            'director': 'Rian Johnson',
            'main_actor': 'Daniel Craig',
            'revenue': 309232797,
        },
        {
            'id': 138,
            'title': "Ocean's Eleven",
            'poster': "https://image.tmdb.org/t/p/original/v5D7K4EHuQHFSjveq8LGxdSfrGS.jpg",
            "overview": "Less than 24 hours into his parole, charismatic thief Danny Ocean is already rolling out his "
                        "next plan: In one night, Danny's hand-picked crew of specialists will attempt to steal more "
                        "than $150 million from three Las Vegas casinos. But to score the cash, Danny risks his "
                        "chances of reconciling with ex-wife, Tess.",
            "genre": "Crime, Thriller",
            "director": "Steven Soderbergh",
            "release_year": "2001",
            "main_actor": "George Clooney",
            'revenue': 450717150,
        }
    ]
