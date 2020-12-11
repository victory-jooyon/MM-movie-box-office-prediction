NSML = nsml run -e main.py --memory 5G --shm-size 1G -g 1 -c 2 --gpu-model P40 -a
SEED = 52




OPTIONS = --SEED $(SEED)

all:
	$(NSML) "$(OPTIONS)"
