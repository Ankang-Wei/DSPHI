   ## Msfei

A new framework based on logical probability theory and network discretization prediction.

   #### Developer

An Kangwei and Xing Pengjiang, School of Mathematics and Statistics, Central China Normal University.

   #### data set

   -data /V.csv: Cosine similarity of phagosomes.
   -data /h.csv: Cosine similarity of hosts.
   -data /high3-JH-HB.csv is the mutual information of type 3 phages after screening and merging.
   -data /high, csv is the mutual information of the host.
   -data /VH.csv: Conference Hosting Association. If the thallus is associated with the host, its label is 1. Otherwise, the label will be 0.
   -data /high1.csv is the mutual information for type 1 phage.
   -data /high2.csv is the mutual information for type 2 phage.
   -data /high3.csv is the mutual information of type 3 phage.
   -data /high4.csv is the mutual information for type 4 phage.
   -data /high5.csv is the mutual information of type 5 phage.
   -data /high6.csv is the mutual message of type 6 phagosomes.
   -data /high7.csv is the mutual message of type 7 phagosomes.
   -data /high8.csv is the mutual information of type 8 phagosome.

   #### coding

   #### tool

When annotating metagenomic data, users can refer to Kneadata [ https://github.com/bioenergy-institute/kneadata ] kraken2 [: //github.com/derrickwood/kraken2 ] for tools Download and install.

The dataset used by Kneadata in its host removal process is human.
You can also use a custom dataset for this. Detailed instructions for constructing the dataset are provided at [ HTPS://jotub.com/BioEnergyInstitute/Kneadata ] .

```
Kneadata_Database-Download Human Genome Package 2
kneadata-i1/residential/q1.fstq.gz-i2/residential/q2.fstq.gz-o emissions-50p50-db/residential/human
```

 在注释过程中,克拉肯2使用标准的数据库,其尺寸为55GB,包括重新生成的古细菌、细菌、病毒、质粒、人1和单核。用户也可以使用自定义数据库;关于如何构造数据库的说明,请参阅 [ //吉特布网/德里克伍德/克拉肯2 ] .资料库可於 [ https://benlangmead.github.io/aws-indexes/k2 ] .布雷肯和克拉肯2都使用相同的数据库。

```
Kraken 2----Database/Home/Standards----Thread 20----Report Marker----Report Test.
Report, report, report, report, report, report, report, report, report, report, report, report, report, report, report, report, report, report, report, report, report.
```

The package used to calculate logical relationships is written in C++, and users can directly call the file used by **L.CPP**.

  ##### Environmental requirements

The required packages are as follows:

  - Python == 3.8.3
  - Hard == 2.8.0
  - Tensorflow == 2.3.0
 - Numpy == 1.23.5
 - Pandas == 1.5.3
 - Protobuf == 3.20.3

 ##### usage

```
GIT clone HTPS://gitub.com/伟康康258369/MSFP
encoding/encoding
large giant snake
```

Users can train predictive models using their own data.

For **new hosts/phages**, users can download DNA from the NBS database and use the code/signature.

** Note: **

In code/features.py, users need to install the iLearn tool [https://ilearn.erc.monash.edu/ or https://github.com/Superzchen/iLearn] and prepare .fasta file, this file is DNA sequences of all phages/hosts. (when you use iLearn to compute the DNA features, you should set the parameters k of Kmer as 3.)

Then users use main.py to predict PHI.


#### Contact

Please feel free to contact us if you need any help.
