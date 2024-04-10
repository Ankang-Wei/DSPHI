   ## 姆斯菲

基于逻辑概率理论和网络离散化预测的新框架。

   #### 开发者

华中师范大学数学与统计学院安康威和兴鹏江。

   #### 数据集

   - 数据/V.ccv:吞噬体的余弦相似性。
   - 数据/h.cv:主机的余弦相似性。
   - 数据/HH3-JH-HP.cv是筛选和合并后的第3型噬菌体的相互信息。
   - 数据/高,ccv是主机的相互信息。
   - 数据/VH.CSV:会议主办协会。如果菌体与宿主相关联,则其标签为1。否则,标签将是0。
   - 数据/HHH1.ccv是第1型噬菌体的相互信息。
   - 数据/HEH2.ccv是第2型噬菌体的相互信息。
   - 数据/HHH3.ccv是第3型噬菌体的相互信息。
   - 数据/HHH4.ccv是4型噬菌体的相互信息。
   - 5.ccv是第5型噬菌体的相互信息。
   - 6.ccv是6型吞噬体的相互信息。
   - 7.ccv是7型吞噬体的相互信息。
   - 数据/HHH8.ccv是8型吞噬体的相互信息。

   #### 编码

   #### 工具

 当注释宏基因组数据时,用户可以参考Kneadata [ HTPS://焦图布网/生物能源研究所/Kneadata ] 克拉肯2 [ //吉特布网/德里克伍德/克拉肯2 ] 用于工具下载和安装。

Kneadata在去除寄主过程中使用的数据集是人类的。
 您也可以为此使用自定义数据集。构造数据集的详细指引载于 [ HTPS://焦图布网/生物能源研究所/Kneadata ] .

```
Kneadata_数据库-下载人类基因组包2
Kneadata-I1/住宅/Q1.fstq.Gz-I2/住宅/Q2.fstq.Gz-o排放量-50p50-db/住宅/人类
```

 在注释过程中,克拉肯2使用标准的数据库,其尺寸为55GB,包括重新生成的古细菌、细菌、病毒、质粒、人1和单核。用户也可以使用自定义数据库;关于如何构造数据库的说明,请参阅 [ //吉特布网/德里克伍德/克拉肯2 ] .资料库可於 [ https://benlangmead.github.io/aws-indexes/k2 ] .布雷肯和克拉肯2都使用相同的数据库。

```
克拉肯2----数据库/家庭/标准----线程20----报告标记----报告测试。
报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告,报告。
```

用于计算逻辑关系的包是用C++编写的,用户可以直接调用 ** L.CPP ** 使用的文件。

  ##### 环境要求

所需包裹如下:

  - Python == 3.8.3
  - Keras == 2.8.0
  - Tensorflow == 2.3.0
 - Numpy == 1.23.5
 - Pandas == 1.5.3
 - Protobuf == 3.20.3

 ##### 用法

```
GIT克隆人HTPS://吉图布.com/威康康258369/MSFP
编码/编码
大型巨蛇
```

 用户可以使用他们的 ** 自己的数据 ** 培训预测模型。

 为了 ** 新宿主/噬菌体 ** ,用户可以从国家统计局的数据库中下载DNA,并使用代码/特征。

 ** 注: ** 

In code/features.py, users need to install the iLearn tool [https://ilearn.erc.monash.edu/ or https://github.com/Superzchen/iLearn] and prepare .fasta file, this file is DNA sequences of all phages/hosts. (when you use iLearn to compute the DNA features, you should set the parameters k of Kmer as 3.)

Then users use main.py to predict PHI.


#### Contact

Please feel free to contact us if you need any help.
