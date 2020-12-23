# Composing Text and Image for Image Retrieval
Project at Multimedia Information Retrieval Course

# Introduction
Our project is an attempt to use TIRG method with Fashion200k dataset.

This repository is forked from [TIRG](https://github.com/google/tirg).

We have added LSH(Locality Sensitive Hashing) to impove speed retrieval.

[Presetation](https://docs.google.com/presentation/d/1Ga_terlOKyy3bl4hNvTKnTnNhqMD2kNOieVN40JrBqQ/edit#slide=id.gb1a04fa9d2_1_28)

# Implementation

## Setup
- pytorch==1.2.0
- torchvision==0.4.0
- Pillow=5.2.0
- tensorboardX

## Decription code
Almost code is based on [TIRG](https://github.com/google/tirg).

Our code:

- `Compute_recall.ipynb`: Compute recall when apply LSH and no LSH. Then compare result. 

- `Train.ipynb`: Continuous training pretrained model from [TIRG](https://github.com/google/tirg). 
    - [TIRG's pretrained model](https://storage.googleapis.com/image_retrieval_css/pretrained_models/checkpoint_fashion200k.pth).
    - [Our model](https://drive.google.com/file/d/1-JphJLv9lTLr9MC3KyMBVM4NlI-ybSWW/view?usp=sharing)

- `Retrieve_example.ipynb`: Retrieve an example query.

- `LSH.py`: LSH class to create hash table, compute hash value.

- `compute_and_save.py`: Pre-compute all features vector for convinient and speed up retrieval.

## Dataset: Fashion200k dataset
Download the dataset from this [external website](https://github.com/xthan/fashion-200k) Download our generated test_queries.txt from [here](https://storage.googleapis.com/image_retrieval_css/test_queries.txt).

Make sure the dataset include these files:

```
<dataset_path>/labels/*.txt
<dataset_path>/women/<category>/<caption>/<id>/*.jpeg
<dataset_path>/test_queries.txt`
```
Run training & testing:

```
python main.py --dataset=fashion200k --dataset_path=./Fashion200k \
  --num_iters=160000 --model=concat --loss=soft_triplet \
  --learning_rate_decay_frequency=50000 --comment=f200k_concat

python main.py --dataset=fashion200k --dataset_path=./Fashion200k \
  --num_iters=160000 --model=tirg --loss=soft_triplet \
  --learning_rate_decay_frequency=50000 --comment=f200k_tirg
```

## LSH (Locality Sensitive Hashing)
- We implement LSH follow this [link](https://towardsdatascience.com/locality-sensitive-hashing-for-music-search-f2f1940ace23)

- All images feature in database are hashing into 100 spatial search.

- Compare result when retrieve random 5000 queries over 5 times:

![Compare result](images/compare_result.png)

## Retrieve an example

![Example](images/retrieve_example.png)