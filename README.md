# BiBERT-R-Drop
This repository contains code of hybrid model consisting of two different solutions to the Machine Translation problem:
- pre-trained language model: BiBERT ("[BERT, mBERT, or BiBERT? A Study on Contextualized Embeddings for Neural Machine Translation](https://aclanthology.org/2021.emnlp-main.534/)")
- regularization method: R-Drop ("[R-drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)")

## Prerequisites
### Conda environment
```
conda create -n bibert-r-drop python=3.7
conda activate bibert-r-drop
conda install -c conda-forge cudatoolkit=11.3 cudnn=8.2.0
conda install pytorch -c pytorch
conda install -y -c conda-forge tensorboard
```
### Additional packages
```
pip install tensorboardX
pip install transformers
pip install sacremoses
pip install sacrebleu==1.5.1
pip install --editable ./
```

## Preprocessing
Download and prepare IWSLT'14 dataset
```
cd download_prepare
bash download_and_prepare_data.sh
```
After download and preprocessing, three preprocessed data bin will be shown in `download_prepare` folder:
* `data`: de->en preprocessed data for ordinary one-way translation
* `data_mixed`: dual-directional translation data
* `data_mixed_ft`: after dual-directional training, fine-tuning on one-way translation data

### Training
Train a model for for dual-directional translation and further fine-tuning:
```
bash train-dual.sh
```
## Evaluation
Translation for dual-directional model:
```
bash generate-dual.sh
```