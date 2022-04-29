Original Repository: https://github.com/biplob1ly/TransICD

# Instructions to reproduce results:
1. Download and extract MIMIC-III data
2. Place the following files in the "mimicdata" folder: NOTEEVENTS.csv, DIAGNOSES_ICD.csv, PROCEDURES_ICD.csv, D_ICD_DIAGNOSES.csv, D_ICD_PROCEDURES.csv
3. Install the following dependencies: torch, numpy, tensorboard, sklearn, pandas, nltk, IPython, gensim
4. Download the stopwords corpus from NLTK: python -m nltk.downloader stopwords
5. Run the data preprocessor: python preprocessor.py
6. Train the model: 
    - python main.py --model TransICD --batch_size 4
    - python main.py --model Transformer --batch_size 4


Changes to prevent errors:
1. preprocessor.py line 230: change "size" input of word2vec to "vector_size":
2. preprocessor.py line 244: change "index2word" to "index_to_key": embed_size = len(wv.word_vec(wv.index2word[0])) -> embed_size = len(wv.word_vec(wv.index_to_key[0]))


--------------------------------------------------------------------------------



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



