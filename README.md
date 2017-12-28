# NN4NLP

## Members
- Rama Kumar Pasumarthi (ramakumar1729)
- Evangelia Spiliopoulou (spilioeve)
- Hiroaki Hayashi (rooa)

## How to run

1. Install requirements.
2. Run as follows
```sh
python -m src.train \
    --save-dir exp1 \  # Unique directory name.
    --batch-size 16 \
    --embed-dim 100 \
    --tolerate 5 \
    --embedding-file PATH/TO/GLOVE/EMBEDDINGS \ # If using pretrained emb
    --cuda # If using gpu
```

## What's saved in the `--save-dir`

- `model_best.pt` : PyTorch model.
- `train.log` : Training/Dev loss and F-1 history.

## Running relation classifier model.

python run.py train relation_classifier/experiments/concat_reprs.json -s /tmp/relation_classifier_model


## Running tests. (Install pytest)
py.test
