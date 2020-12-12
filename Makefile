NSML = nsml run -e main.py --memory 12G --shm-size 1G -g 1 -c 5 --gpu-model P40 -a
SEED = 4
AUG = max-only

# normal pool-vec allow-grad more-layer
OPTIONS = --SEED $(SEED) --aug $(AUG)

all:
	$(NSML) "$(OPTIONS)"
	$(NSML) "$(OPTIONS) --ablation imdb"
	$(NSML) "$(OPTIONS) --ablation poster"
	$(NSML) "$(OPTIONS) --ablation tmdb"

debug:
	python3 main.py $(OPTIONS) --batch_size 4 --ablation imdb

one:
	$(NSML) "$(OPTIONS)"

imdb:
	$(NSML) "$(OPTIONS) --ablation imdb"

normal:
	$(NSML) "--seed $(SEED) --aug normal --num_classes 3"
	$(NSML) "--seed $(SEED) --aug normal --ablation imdb --num_classes 3"
	$(NSML) "--seed $(SEED) --aug normal --ablation poster --num_classes 3"
	$(NSML) "--seed $(SEED) --aug normal --ablation tmdb --num_classes 3"

pool-vec:
	$(NSML) "--seed 0 --aug pool-vec --num_classes 3"
	$(NSML) "--seed 1 --aug pool-vec --num_classes 3"
	$(NSML) "--seed 2 --aug pool-vec --num_classes 3"
	$(NSML) "--seed 3 --aug pool-vec --num_classes 3"
	$(NSML) "--seed 4 --aug pool-vec --num_classes 3"

pool-max:
	$(NSML) "--seed 52 --aug pool-max"
	$(NSML) "--seed 0 --aug pool-max"
	$(NSML) "--seed 1 --aug pool-max"
	$(NSML) "--seed 2 --aug pool-max"
	$(NSML) "--seed 3 --aug pool-max"
	$(NSML) "--seed 4 --aug pool-max"

max-only:
	$(NSML) "--seed 52 --aug max-only --num_classes 3"
	$(NSML) "--seed 0 --aug max-only --num_classes 3"
	$(NSML) "--seed 1 --aug max-only --num_classes 3"
	$(NSML) "--seed 2 --aug max-only --num_classes 3"
	$(NSML) "--seed 3 --aug max-only --num_classes 3"
	$(NSML) "--seed 4 --aug max-only --num_classes 3"

mlp:
	$(NSML) "--seed 0 --aug mlp"
	$(NSML) "--seed 0 --aug mlp --ablation imdb"
	$(NSML) "--seed 1 --aug mlp"
	$(NSML) "--seed 1 --aug mlp --ablation imdb"
	$(NSML) "--seed 2 --aug mlp"
	$(NSML) "--seed 2 --aug mlp --ablation imdb"
	$(NSML) "--seed 3 --aug mlp"
	$(NSML) "--seed 3 --aug mlp --ablation imdb"
	$(NSML) "--seed 4 --aug mlp"
	$(NSML) "--seed 4 --aug mlp --ablation imdb"

test:
	$(NSML) "--seed 52 --aug mlp"
	$(NSML) "--seed 52 --aug mlp --ablation imdb"