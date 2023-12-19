# SpaCEX: A self-supervised learning on spatially co-expressed genes in spatial transcriptomics data
We develop the **SpaCEX** which utilize self-supervised learning on **Spa**tially *C*o-**EX**pressed genes that can simultaneously identify spatially co-expressed genes and learn semantically meaningful gene embeddings from SRT data through a pretext task of gene clustering. **SpaCEX** first employs an image encoder to transform the spatial expression maps of genes into gene embeddings modeled by a Student’s t mixture distribution (SMM). Subsequently, a discriminatively boosted gene clustering algorithm is applied on the posterior soft assignments of genes to the mixture components, iteratively adapting the parameters of the encoder and the SMM. 
# Overview of SpaCEX

<p align="center">
  <img src="https://github.com/WLatSunLab/SpaCEX/assets/121435520/78eb358a-70d5-4036-bc08-85ff0041b8bc" width="900">
</p>



# Framework of SpaCEX
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
* Enhancement of the transcriptomic coverage.
* Identify spatially co-expressed and co-functional genes.
* Predict gene-gene interactions.
* Detect spatially variable genes.
* Cluster spatial spots into tissue domains.

# Installation
You can download the package from GitHub and install it locally:
```bash
git clone https://github.com/WLatSunLab/SpaCEX.git
```
# Sample data
Sample data include 10x-hDLPFC-151676, 10x-mEmb, seq-mEmb can be fund [here](https://drive.google.com/drive/folders/1C3Gk-HVYp2dQh4id8H68M9p8IWEOIut_? usp=drive_link). and make sure these data are organized in the following structure:
-SpaCEX
  -s
# Getting Started
The [tutorial](SpaCEX_ETC.ipynb) included in the repository provides guidance on how to effectively utilize SpaCEX.
# Others
The tutorial will continue to receive ongoing updates for a more detailed and comprehensive guide, along with an refined model.

