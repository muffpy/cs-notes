# 6.047/6.878/HST.507 Fall 2020 Prof. Manolis Kellis
Computational Biology: Genomes, Networks, Evolution, Health

Machine Learning in Genomics: Dissecting the circuitry of Human Disease

## Why computational biology?

Biological systems are fundamentally digital in nature with information encoded in four letters: A, T, G and C in the form of DNA. These codes can be stored, replicated and processed.

New technologies such as sequencing and high-throughput experimental techniques like microarray, yeast two-hybrid, single cell RNA-seq, and ChIP-chip assays are creating enormous and increasing amounts of data waiting to be analysed.

Computational progress in terms of processing power, storage capacity, network capacity and advances in algorithmic techniques and machine learning has made data processing more efficient.

Biological datasets are noisy and identifying robust signals from noise is an inherently computational problem.
- the biological signals we are interested in are called functional elements of DNA, eg. protein-coding regions, promoters, ehancers, regulatory motifs, enhancers,...
- How do we identify these regions of DNA from non-fucntional regions?

## Gene sequence alignment
Sequence alignment is a powerful tool that assesses the similarity of sequences in order to learn about
their function or their evolutionary relationship. If two genetic regions are similar or identical, sequence
alignment can demonstrate the conserved elements or differences between them. 