# SLSCG: Self-supervised Learning on Spatially Co-expressed Genes
We develop the **SLSCG** (**S**elf-supervised **L**earning on **S**patially **C**o-expressed **G**enes ) model that can simultaneously identify spatially co-expressed genes and learn semantically meaningful gene embeddings from SRT data through a pretext task of gene clustering. **SLSCG** first employs an image encoder to transform the spatial expression maps of genes into gene embeddings modeled by a Student’s t mixture distribution (SMM). Subsequently, a discriminatively boosted gene clustering algorithm is applied on the posterior soft assignments of genes to the mixture components, iteratively adapting the parameters of the encoder and the SMM. 
<p align="center">
  <img src="https://github.com/image-deep-clustering/SLSCG/assets/121435520/4609bb4b-452e-4889-a21d-ad0753d0f55c" width="800">
</p>


# Applicable tasks
* Identify spatially co-expressed and co-functional genes.
* Predict gene-gene interactions.
* Detect spatially variable genes.
* Cluster spatial spots into tissue domains
