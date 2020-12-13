# MM-movie-box-office-prediction
KAIST CS470 Team 24 Final Project

- 20130240 박규민
- 20160156 김주연
- 20160276 박한길
- 20160413 윤소영

## Run
### Run Multi-modal
For runnin gour multi-feature model, the below command is enough.
```shell script
python main.py
```
If you want to customize the model by changing hyperparameters, you are able to give more arguments. (epochs, batch_size, valid_interval, lr, num_workers, device, seed, etc)

The example is as below.
```shell script
python main.py --epochs 20 --batch_size 20 --lr 0.000001 
```

### Run Single-feature model
```shell script
python main.py --ablation {feature}
```
Single features include poster / tmdb (overview) / imdb.

You are able to change the hyperparameters with the same way explained in the multi-modal part.

## Data
This step is not mandatory, because we uploaded the data file for you. But if you want to reproduce from collecting the data by your own, you can follow the steps before running the model.

1. Run [this script](https://github.com/victory-jooyon/MM-movie-box-office-prediction/blob/main/data/scripts/crawl_imdb.py), after reading the comments from the file. This aggregates the tsv files given from IMDB. Or you can pass this step by simply downloading the file from [here](https://drive.google.com/file/d/1vc5kDLmuFc8G4DChHhDJFXgLlZ68wu_i/view?usp=sharing) and put it under MM-movie-box-office-prediction/data/json/.
2. Relate TMDB data with the IMDB data from STEP1, using [this script](https://github.com/victory-jooyon/MM-movie-box-office-prediction/blob/main/data/scripts/crawl_all.py). To run this script, you need to signup to [TMDB](https://www.themoviedb.org/?language=ko), and then have your own API key as the value for environment variable `TMDB_API_KEY`. Your can simply do this by command `export TMDB_API_KEY={your own TMDB API key}`
3. And then aggregate the data collected from STEP2, using [this script](https://github.com/victory-jooyon/MM-movie-box-office-prediction/blob/main/data/scripts/preprocess_crawled_data.py).

After you are done with the three steps, you will have your own [crawled_data_all.json](https://github.com/victory-jooyon/MM-movie-box-office-prediction/blob/main/data/json/crawled_data/crawled_data_all.json), and you are ready for running the model.
