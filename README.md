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
    --data-dir data/processed \
    --embedding-file PATH/TO/GLOVE/EMBEDDINGS \ # If using pretrained emb
    --cuda # If using gpu
```
