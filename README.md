## DCA
DCA (Dynamic Context Augmentation) provides global entity linking models featuring:

- **Efficiency** __Strong__: Comparing to global entity linking models, DCA only requires one pass through all mentions, yielding better efficiency in inference.

- **Portability** __Strong__: DCA can introduce topical coherence into local linking models without reshaping their original designs or structures.

Details about DCA can be accessed at: https://arxiv.org/abs/1909.02117.

This implementation refers to the project structure of [mulrel-nel](https://github.com/lephong/mulrel-nel).

## Data
Download [data](https://drive.google.com/file/d/1xW-t80cKDMx3ZL-hrRUxlm6QMZIRvUyU/view) from here and unzip to the main folder (i.e. your-path/DCA). 

## Installation

Requirements: Python 3.5 or 3.6, Pytorch 0.3, CUDA 7.5 or 8

## Usage
cd DCA

export PYTHONPATH=$PYTHONPATH:../

SL: python mulrel_ranker.py --mode train --order offset --model_path model --method SL

RL: python mulrel_ranker.py --mode train --order offset --model_path model --method RL
