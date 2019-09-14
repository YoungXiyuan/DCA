## DCA
DCA (Dynamic Context Augmentation) provides global entity linking models featuring:

- **Efficiency**: Comparing to global entity linking models, DCA only requires one pass through all mentions, yielding better efficiency in inference.

- **Portability**: DCA can introduce topical coherence into local linking models without reshaping their original designs or structures.

Remarkablely, our DCA models (trained by supervised learning or reinforcement learning) achieved:

- **94.64%** in-KB acc. on AIDA-CONLL testset (AIDA-B).
- **94.57%** F1 score on MSBNC dataset and **90.14%** F1 score on ACE2004 dataset.

Details about DCA can be accessed at: https://arxiv.org/abs/1909.02117.

This implementation refers to the project structure of [mulrel-nel](https://github.com/lephong/mulrel-nel).

## Overall Workflow
![Alt Text](https://github.com/YoungXiyuan/DCA/blob/master/DCA.gif)

## Data
Download [data](https://drive.google.com/file/d/1xW-t80cKDMx3ZL-hrRUxlm6QMZIRvUyU/view) from here and unzip to the main folder (i.e. your-path/DCA). 

The above data archive mainly contains the following resource files:

- **Dataset**: One in-domain dataset (AIDA-CoNLL) and Five cross-domain datasets (MSNBC / AQUAINT / ACE2004 / CWEB / WIKI). And these datasets share the same data format.

- **Type Embedding**: Adopted to compute type similarity between mention-entity pairs. We trained these type embedding using a typing system called [NFETC](https://arxiv.org/abs/1803.03378) model.

- **Wikipedia inLinks**: Surface names of inlinks for a Wikipedia page (entity) are used to construct **dynamic context** in our model learning process.

## Installation
Requirements: Python 3.5 or 3.6, Pytorch 0.3, CUDA 7.5 or 8

## Important Parameters

```
mode: train or eval mode.

method: training method, Supervised Learning (SL) or Reinforcement Learning (RL)

order: three decision orders -- offset / size / random. Please refer to our paper for their concrete definition.

n_cands_before_rank: the number of candidates, the default value is 30.

tok_top_n4inlink: the number of inlinks for a Wikipedia page (entity) would be considered as candidates for the dynamic context.

tok_top_n4ent: the number of inlinks for a Wikipedia page (entity) would be added into the dynamic context.

isDynamic: 2-hop DCA / 1-hop DCA / without DCA. Corresponding to the Table 4 in our paper.

dca_method: soft+hard attention / soft attention / average sum. Corresponding to the Table 5 in our paper.
```

## Running
cd DCA/

export PYTHONPATH=$PYTHONPATH:../

Supervised Learning: python main.py --mode train --order offset --model_path model --method SL

Reinforcement Learning: python main.py --mode train --order offset --model_path model --method RL

## Citation
If you find the implementation useful, please cite the following paper: [Learning Dynamic Context Augmentation for Global Entity Linking](https://arxiv.org/abs/1909.02117)

```
@article{yang2019learning,
  title={Learning Dynamic Context Augmentation for Global Entity Linking},
  author={Yang, Xiyuan and Gu, Xiaotao and Lin, Sheng and Tang, Siliang and Zhuang, Yueting and Wu, Fei and Chen, Zhigang and Hu, Guoping and Ren, Xiang},
  booktitle = {EMNLP},
  year={2019}
}
```
