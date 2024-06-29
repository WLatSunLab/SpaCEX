# SpaCEX: A self-supervised learning on spatially co-expressed genes in spatial transcriptomics data
We develop the **SpaCEX** which utilize self-supervised learning on spatially co-expressed genes that can simultaneously identify spatially co-expressed genes and learn semantically meaningful gene embeddings from SRT data through a pretext task of gene clustering. **SpaCEX** first employs an image encoder to transform the spatial expression maps of genes into gene embeddings modeled by a Student’s t mixture distribution (SMM). Subsequently, a discriminatively boosted gene clustering algorithm is applied on the posterior soft assignments of genes to the mixture components, iteratively adapting the parameters of the encoder and the SMM. 
<p align="center">
  <img src="https://github.com/Shaw-Lab/SpaCEX/assets/121435520/97d5e386-5606-49a3-8b7e-4a3b1a921c4e.png" width="900">
</p>

# Dependencies
* Python = 3.9.15
* torch = 1.13.0
* sklearn = 1.2.0
* scanpy = 1.9.3
* scipy = 1.9.3
* pandas = 1.5.2
* numpy = 1.21.6
* sympy = 1.11.1


# Applicable tasks
* Identify spatially co-expressed and co-functional genes.
* Predict gene-gene interactions.
* Detect spatially variable genes.
* Cluster spatial spots into tissue domains

# Installation
You can download the package from GitHub and install it locally:
```bash
git clone https://github.com/WLatSunLab/SpaCEX.git
cd SpaCEX
```

# Getting Started
**What** you should input:

`dataset` gene image data with size [N, 1, 72, 59]

`total` gene similarity matrix with size [N, N]

**What** you will get return:

`model` encoder that haved been trained

`y_pred` label that SpaCEX generative

`embedding` embedding that generatived by encoder
```bash
from SpaCEX.src.main.SpaCEX import SpaCEX

y_pred, embedding, model = SpaCEX.train(dataset = dataset, total = total)
```
# Others
If you wanna get other return such as x_bar or parameters of SMM, just rewrite DEC to get what you want.

