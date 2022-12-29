# MetaAc4C
MetaAc4C: A multi-module deep learning framework for accurate prediction of N4-acetylcytidine sites based on pre-trained bidirectional encoder representation and generative adversarial networks.


The source code and datasets(both training and testing datasets) can be freely download from the github

## Brife tutorial

### 1. Environment requirements
Before running, please make sure the following packages are installed in Python environment:

gensim==3.4.0  
pandas==1.0.3  
tensorflow==2.3.0  
python==3.8.8  
biopython==1.7.8  
numpy==1.19.2  
torch==1.9.1

For convenience, we strongly recommended users to install the Anaconda Python 3.8.8 (or above) in your local computer.

### 2. Running BERT
Changing working dir to MetaAc4C, and then running the following Build bert environment according to bert requirements：tf1_py2
https://github.com/google-research/bert

command:  
source activate tf1_py2
CUDA_VISIBLE_DEVICES=2 python /BERT-All_models/codes/extract-last-layer-features_All_config.py -species 'ac4c' -max-len 415

-species: input data
-max-len: the max length of squences

### 3. Runing MetaAc4C
python MetaAc4c_code.py


