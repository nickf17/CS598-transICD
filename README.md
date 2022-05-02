

# Instructions to reproduce results:
Original code: https://github.com/biplob1ly/TransICD
1. Download and extract MIMIC-III data
2. Place the following files in the "mimicdata" folder: NOTEEVENTS.csv, DIAGNOSES_ICD.csv, PROCEDURES_ICD.csv, D_ICD_DIAGNOSES.csv, D_ICD_PROCEDURES.csv
3. Install the following dependencies: torch, numpy, tensorboard, sklearn, pandas, nltk, IPython, gensim
4. Download the stopwords corpus from NLTK: python -m nltk.downloader stopwords
5. Perform data preprocessing: python preprocessor.py
6. Train a new model: 
    - TransICD: python main.py --model TransICD --batch_size 4
    - Transformer: python main.py --model Transformer --batch_size 4
    - Transformer + Attention (no LDAM loss): 
       1. replace lines 178-181 in models.py with "ldam_outputs = None" 
       2. python main.py --model TransICD --batch_size 4
    - Biaffine Transformation ablation:
       1. in main.py: replace "from models import *" with "from model_ablation import *"
       2. python main.py --model TransICD --batch_size 4

Changes to prevent errors:
1. preprocessor.py line 230: change "size" input of word2vec to "vector_size"
2. preprocessor.py line 244: change "index2word" to "index_to_key": embed_size = len(wv.word_vec(wv.index2word[0])) -> embed_size = len(wv.word_vec(wv.index_to_key[0]))


--------------------------------------------------------------------------------
# Instructions to reproduce baselines (LR, Bi-GRU, CNN, CAML):
Original code: https://github.com/jamesmullenbach/caml-mimic


*all instructions are with respect to the "/baselines/" folder
1. Download and extract MIMIC-III data
2. Place the following files in the "mimicdata" folder: D_ICD_DIAGNOSES.csv, D_ICD_PROCEDURES.csv
3. Place the following files in the "mimicdata/mimic3" folder: NOTEEVENTS.csv, DIAGNOSES_ICD.csv, PROCEDURES_ICD.csv
4. Install the following dependencies: torch, numpy, sklearn, pandas, nltk, gensim, scipy, jupyter-notebook, tqdm
5. Modify constants.py: set DATA_DIR='../mimicdata' and MIMIC_3_DIR='../mimicdata/mimic3'
6. Perform data preprocessing: run all cells in notebooks/dataproc_mimic_III.ipynb
7. Train a new model:


-Logistic Regression (predictions/LR_mimic3_50):


```python ../../learn/training.py ../../mimicdata/mimic3/train_50.csv ../../mimicdata/mimic3/vocab.csv 50 logreg 100 --pool avg --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu```


-Bidirectional GRU (predictions/GRU_mimic3_50):


```python ../../learn/training.py ../../mimicdata/mimic3/train_50.csv ../../mimicdata/mimic3/vocab.csv 50 rnn 200 --cell-type gru --rnn-dim 512 --bidirectional True --lr 0.003 --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu```


-CNN (predictions/CNN_mimic3_50):


```python ../../learn/training.py ../../mimicdata/mimic3/train_50.csv ../../mimicdata/mimic3/vocab.csv 50 cnn_vanilla 100 --filter-size 4 --num-filter-maps 500 --dropout 0.2 --lr 0.003 --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu```


-CAML (predictions/caml_mimic3_50):


```python ../../learn/training.py ../../mimicdata/mimic3/train_50.csv ../../mimicdata/mimic3/vocab.csv 50 conv_attn 200 --filter-size 10 --num-filter-maps 50 --dropout 0.2 --patience 10 --criterion prec_at_8 --lr 0.0001 --embed-file ../../mimicdata/mimic3/processed_full.embed --gpu```


Changes to prevent errors:
1. word_embeddings.py line 25: change "size" input of word2vec to "vector_size"
2. word_embeddings.py line 25: change "iter" input of word2vec to "epochs"
3. word_embeddings.py line 30: change value of epochs input from "model.iter" to "model.epochs"
4. extract_wvs.py line 42 and 44: replace "index2word" with "index_to_key"
5. training.py line 40: import ctypes and change csv.field_size_limit(sys.maxsize) to csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
6. training.py line 207 and 271: change losses.append(loss.data[0]) to losses.append(loss.item())
7. models.py line 282 and 284: change bidirectional=bidirectional to bidirectional=bool(bidirectional)
8. models.py line 121: remove the line x = x.transpose(1, 2)
9. models.py line 128: change x = F.avg_pool1d(x) to x = torch.mean(x, 1)
10. models.py line 131: change "diffs" input of self._get_loss to "desc_data"
11. models.py line 132: change return value "yhat" to "logits"
--------------------------------------------------------------------------------


# Reproduced Results:


```
---------------------------------------------------------------------------------------------------------------
|                                               |            AUC            |           F1        |           |
|Model                                          |-------------|-------------|-----------|---------|   P@5     |
|                                               |      Macro  |    Micro    |  Macro    |  Micro  |           |
|-----------------------------------------------|-------------------------------------------------------------|
|LR                                             |      49.2        59.6        21.6        21.6       27.9    |
|Bi-GRU                                         |      79.2        84.3        41.2        51.3       50.4    |
|CNN                                            |      87.6        90.2        57.9        62.4       61.2    |
|CAML                                           |      85.6        88.0        50.7        56.3       55.4    |
|-----------------------------------------------|-------------------------------------------------------------|
|Transformer                                    |      85.0        89.2        39.0        54.2       56.1    |
|Transformer + Attention                        |      89.5        92.4        51.1        62.0       61.4    |
|TransICD (Transformer + Attention + LDAM_loss) |   88.6+\-0.4  92.0+\-0.3  54.5+\-0.5  63.8+\-0.5 61.2+\-0.4 |
---------------------------------------------------------------------------------------------------------------
```

# Ablation study results:
```
---------------------------------------------------------------------------------------------------------------
|                                               |            AUC            |           F1        |           |
|Model                                          |-------------|-------------|-----------|---------|   P@5     |
|                                               |      Macro  |    Micro    |  Macro    |  Micro  |           |
|-----------------------------------------------|-------------------------------------------------------------|
|TransICD + code-independent representation     |      87.1        90.5        53.7        61.1       58.3    |
---------------------------------------------------------------------------------------------------------------
```
