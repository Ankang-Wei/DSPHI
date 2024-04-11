## DSPHI

Code and Datasets for "MSPHI: A novel framework for phage-host prediction via logical probability theory and network sparsification"

#### Developers

Ankang Wei (weiankng@mails.ccun.edu.cn) and Xingpeng Jiang (xpjiang@mail.ccnu.edu.cn) from School of Mathematics and Statistics, Central China Normal University.

#### Datasets

- data/V.csv: Cosine similarity of phage.
- data/H.csv: Cosine similarity of host.
- data/high3-JH-HB.csv is the mutual information of phage type 3 after filtering and merging.
- data/high-h,csv is the mutual information of host.
- data/VH.csv: phage-host associations. If the phage is associated with host, its label will be 1. Otherwise, the label will be 0.
- data/high1.csv is the mutual information of phage type 1.
- data/high2.csv is the mutual information of phage type 2.
- data/high3.csv is the mutual information of phage type 3.
- data/high4.csv is the mutual information of phage type 4.
- data/high5.csv is the mutual information of phage type 5.
- data/high6.csv is the mutual information of phage type 6.
- data/high7.csv is the mutual information of phage type 7.
- data/high8.csv is the mutual information of phage type 8.

#### Code

#### Tool

When annotating metagenomic data, users can refer to kneaddata [https://github.com/biobakery/kneaddata] and kraken2 [https://github.com/DerrickWood/kraken2] for tool downloading and installation. 

The dataset used by kneaddata in the host removal process is human_hg38_refMrna. 
You can also use a custom dataset for this purpose. Detailed instructions for constructing the dataset can be found at [https://github.com/biobakery/kneaddata].

```
kneaddata_database --download human_genome bowtie2
kneaddata -i1 /home/q1.fastq.gz -i2 /home/q2.fastq.gz -o output_dir -t 50 -p 50 -db /home/human_hg38_refMrna
```

During the annotation process, Kraken2 uses the Standard database, which is 55GB in size and includes RefSeq archaea, bacteria, viral, plasmid, human1, and UniVec_Core. Users can also use a custom database; for instructions on how to construct one, please refer to [https://github.com/DerrickWood/kraken2]. The database can be downloaded from [https://benlangmead.github.io/aws-indexes/k2]. Both Bracken and Kraken2 use the same database.

```
kraken2 --db /home/Standard  --threads 20 --report flag --report TEST.report --output TEST.output  --paired q1.fastq.gz q2.fastq.gz
bracken -d /home/Standard -i TEST.report -o TEST.S.bracken -w TEST.S.bracken.report -r 150 -l S
```

The package used for computing logical relationships is written in C++ and users can directly call the **L.cpp** file for its usage.

##### Environment Requirement

The required packages are as follows:

- Python == 3.8.3
- Keras == 2.8.0
- Tensorflow == 2.3.0
- Numpy == 1.23.5
- Pandas == 1.5.3
- Protobuf == 3.20.3

##### Usage

```
git clone https://github.com/weiankang258369/MSPHI
cd MSPHI/code
python main.py
```

Users can use their **own data** to train prediction models. 

For **new host/phage**, users can download the DNA from the NCBI database, and use code/features.py to compute the features derived from DNA.

**Note:** 

In code/features.py, users need to install the iLearn tool [https://ilearn.erc.monash.edu/ or https://github.com/Superzchen/iLearn] and prepare .fasta file, this file is DNA sequences of all phages/hosts. (when you use iLearn to compute the DNA features, you should set the parameters k of Kmer as 3.)

Then users use main.py to predict PHI.


#### Contact

Please feel free to contact us if you need any help.
