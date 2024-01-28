# SpaCEX: A self-supervised learning on spatially co-expressed genes in spatial transcriptomics data
We develop the **SpaCEX** which utilize self-supervised learning on **Spa**tially **C**o-**EX**pressed genes that can simultaneously identify spatially co-expressed genes and learn semantically meaningful gene embeddings from SRT data through a pretext task of gene clustering. **SpaCEX** first employs an image encoder to transform the spatial expression maps of genes into gene embeddings modeled by a Student’s t mixture distribution (SMM). Subsequently, a discriminatively boosted gene clustering algorithm is applied on the posterior soft assignments of genes to the mixture components, iteratively adapting the parameters of the encoder and the SMM. 
# Overview of SpaCEX

<p align="center">
  <img src="https://github.com/WLatSunLab/SpaCEX/assets/121435520/01cde816-a104-49fe-a875-abc34f6aac1e" width="700">
</p>

# Dependencies
```
[Python 3.9.15]
[torch 1.13.0]
[rpy2 3.5.13]
[sklearn 1.2.0]
[scanpy 1.9.3]
[scipy 1.9.3]
[pandas 1.5.2]
[numpy 1.21.6]
[sympy 1.11.1]
[SpaGCN 1.2.7]
[anndata 0.10.3]
```

# Applicable tasks
```
* Enhancement of the transcriptomic coverage.
* Identify spatially co-expressed and co-functional genes.
* Predict gene-gene interactions.
* Detect spatially variable genes.
* Cluster spatial spots into tissue domains.
```

# Installation
You can download the package from GitHub and install it locally:
```bash
git clone https://github.com/WLatSunLab/SpaCEX.git
```
# Sample data
Sample data including 10x-hDLPFC-151676, 10x-mEmb, seq-mEmb can be found [here](https://drive.google.com/drive/folders/1C3Gk-HVYp2dQh4id8H68M9p8IWEOIut_?usp=drive_link) and make sure these data are organized in the following structure:
```
 . <SpaCEX>
        ├── ...
        ├── <data>
        │   ├── 151676_10xvisium.h5ad
        │   ├── DLPFC_matrix_151676.dat
        │   └── <mEmb>
        │       ├── 10x_mEmb_matrix.dat
        │       ├── sqf_mEmb_adata.h5ad
        │       └── qf_mEmb_matrix.dat
        ├── <model_pretrained>
        │   │
        └── ...

```
# Getting Started
The [tutorial](https://zipging.github.io/-SpaCEX-.github.io/) included in the repository provides guidance on how to effectively utilize SpaCEX.
# Others
The tutorial will continue to receive ongoing updates for a more detailed and comprehensive guide, along with an refined model.

