# Chlamydomonas Reinhardtii RPKM prediction

Using Convolutional Neural Networks to predict gene expression levels based on promoter sequences

## Data 

[one_hot_500.picke](https://www.dropbox.com/s/o4m08hf7ozgerps/onehot500.pickle?dl=0)

[onehot500bp.-450+50.txt](https://www.dropbox.com/s/y70ksolfle2pl06/onehot500bp.-450%2B50.txt?dl=0)

[Zones RPKMs](https://www.dropbox.com/s/evm2hf059xcwrpf/Zones-all%20copy.xlsx?dl=0)

[Chlamydomonas Reinhardtii Genenome Annotation](https://www.dropbox.com/s/2gdu839b7e65frx/Creinhardtii_281_v5.5.gene.gff3?dl=0)

[Clamydomonas Sequenced Genome](https://www.dropbox.com/s/uy54py0nb44vabd/Sample_fasta_input.fasta.csv?dl=0)


## Requirements

This script requires Python 3.x and Tensorflow 1.x

[Tensorflow for Mac](https://www.tensorflow.org/install/install_mac)

[Tensorflow for Windows](https://www.tensorflow.org/install/install_windows)

[Tensorflow for Ubuntu](https://www.tensorflow.org/install/install_linux)


## Run Neural Network

```
python promoter_rpkm_prediction.py
```

## To generate data from scratch

1) Run R code to get sequences from genome -> (*R_output*)

2) Use **processSequence.py** to trim sequence to defined promotmer length

3) Copy gene names and normalized RPKM values to new csv file ->(*PRKM from zones paper*)

4) Run **matchSeqToRPKM.py**  to match *R_output* to *RPKM from Zones paper*

```
python matchSeqToRPKM RPKM_from_Zones_paper.csv
```

5) One hot encode sequence with **onehot_encode2.py**
