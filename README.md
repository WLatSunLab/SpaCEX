# SpaCEX: A self-supervised learning on spatially co-expressed genes in spatial transcriptomics data
We develop the **SpaCEX** which utilize self-supervised learning on spatially co-expressed genes that can simultaneously identify spatially co-expressed genes and learn semantically meaningful gene embeddings from SRT data through a pretext task of gene clustering. **SpaCEX** first employs an image encoder to transform the spatial expression maps of genes into gene embeddings modeled by a Student’s t mixture distribution (SMM). Subsequently, a discriminatively boosted gene clustering algorithm is applied on the posterior soft assignments of genes to the mixture components, iteratively adapting the parameters of the encoder and the SMM. 
<p align="center">
  <img src="https://github.com/Shaw-Lab/SpaCEX/assets/121435520/97d5e386-5606-49a3-8b7e-4a3b1a921c4e.png" width="900">
</p>

# Applicable tasks
* Identify spatially co-expressed and co-functional genes.
* Predict gene-gene interactions.
* Detect spatially variable genes.
* Cluster spatial spots into tissue domains

# Getting Started
```bash
git clone https://github.com/WLatSunLab/SpaCEX.git
cd SpaCEX
```
