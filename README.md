# outline-NLP2023

**这个文件里没有的都不会考**

**凡是留过的作业都可能会考**
**所有的神经网络都不考**
**题型：计算题+解答题**
**简答题示例：简述依存关系**
**不用背，但是需要知道怎么用，但是需要知道基本概念**
**要考的都在这些topics中, 这些以外的都不考**
**例题和留的作业是复习重点**
**课程目标：没有网络也能写NLP程序**
**可能出现手写代码，比如"手写k-means程序", 可以是伪代码, 再举个例子, "把文本分类再做一遍", 评分标准是基本步骤是不是都在，而不是一定要过编译，能跑之类的, 注重的是程序的完整性和程序的有效性**
**上课没讲过的/发的PPT里没有的不会考**


# 语言模型

- n-gram
  - 建模
- 神经网络语言模型(NNLM):

# 词

## 一、词法分析（L4）

- 形态分析(L4 P3-15)
  - 基本任务
    - 单词识别
      - 缩略语的识别
        - prof., Mr., Ms. Co., Oct. （词典）
        - can’t => can + not;
    - 形态还原
      - 有规律变化单词的形态还原
        - -ing, -ed, -s, -ly, -er/est, 名词所有格
      - 不规则
        - 建立不规则变化词表 （词典）
      - 年代、时间、百分数、货币、序数词（规则）
      - 合成词
        - one-fourth序数词
        - machine-readable合成形容词
  - 一般方法（P15）
- 分词（L4 P16-63）
  - 重要性
  - 主要问题
    - 歧义切分字段处理P18-22
    - 未登录词的识别P23-25
  - 基本原则：两个合并原则P26-27
  - 辅助原则P28-34
    - 两个切分原则
    - 四个合并原则
  - 评价方法
    - 评价指标P36-40
      - 正确率
      - 召回率 Recall-ratio
      - F测度和F1
    - 基本算法
      - 最大匹配法(Maximum Matching, MM) P42-46
        - FMM: Fast Maximum Matching
      - 最少分词法(最短路径法) P47-50
      - 基于语言模型的分词方法 P51-52
      - 基于HMM的分词方法 P53-54
      - 由字构词 (基于字标注)的分词方法 P55-56
      - 生成式方法与区分式方法的结合 P57-58
        - 生成式模型与判别式模型的比较 P59-61
      - 方法比较P62-64
  - 未登陆词识别
    - 未登陆词分类 P65-68
      - 命名实体（专有名词）
      - 其他新词
    - 中文姓名识别 P69-76
      - 难点
      - 方法
      - 评估函数
      - 修饰规则
    - 中文地名识别 P77-79
      - 难点
      - 基本资源
      - 基本方法
    - 中文机构名称的识别 P80-83
    - 命名实体的神经网络识别方法 P84-87
      - 基于RNN
      - 基于LSTM

## 二、词性标注 part-of-speech tagging (POS tagging) (L4)

- 概述 P88-94
- 方法
  - Rule-Based (基于规则的方法)
  - Learning-Based (基于学习的方法)
    - 统计模型
      - **隐马尔可夫模型 (HMM) P116-182**
        - 定义 P116-129
        - HMM for ice cream task P130-132
        - 三个基本问题 （机器学习第五章ppt）
          - Problem 1 (估算问题)
            - 计算观测的似然 132-138
            - 前向算法 139-145
            - 后向算法 147-149
          - Problem 2 (解码问题)
            - the Viterbi algorithm 152-158
          - Problem 3 (参数学习)
            - 前向-后向算法 159-168
        - HMM应用于POS Tagging 169-172
      - ~~条件随机域模型 (CRF)~~
    - 规则学习
      - 基于转换的学习 (TBL)
- 其他序列标注任务
  - 命名实体识别
  - 分词

## 三、词义分析 (L5)

- 基于符号的词义分析
  - 借鉴词典的词义定义（寻找基本词（即义素、语义特征、义原）的方法）
    - 专家定义 P11-13
    - 自动发现 P14
  - 利用词间关系定义（词间关系：即两个词的某两个义位之间的关系，主要是词义关系） P18-42
    - 例如有：上下义位关系(Hyponymy)、全体-成员关系(Ensemble-Member)、整体-部分关系(Whole-Part)、同义关系(Synonymy)等关系
    - 词义消歧(WordSense Disambiguation:WSD)的两类具体任务：Lexical Sample任务（某个特定词）、All-words WSD任务（所有目标多义词）
      
- 基于数值（向量）的词义分析
  - （可分为：point embedding（一个N维的实向量）、Gaussian embedding（一个N维的高斯分布））
  -  有3种获得embedding的方法
    - 1、基于全局上下文的统计方法：先建立高维词向量，之后基于降维技术获得低维词向量
    - 2、基于局部上下文的预测方法：直接获得低维词向量的word2vecter(CBOW,SGNG)等。如：**CBOW** + Hierarchical SoftMax、**Skip-Gram** + Negative Sampling
    - 3、二者的结合：如GLoVe
  - 1、分布式词义表示 47-49
    - ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/4a765052-eb79-43fe-8507-6e388d3ef144)
    - 降维 50-51
      - 选择某些维（如TF-IDF）、SVD进行分解、PCA降维
  - 2、基于预测的低维向量
    ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/99c2fdb8-ba0c-46af-8f52-800d5459cfe4)
    - **CBOW** + Hierarchical SoftMax    59-75
      - CBOW：用周围词预测中心词，从而利用中心词的预测结果情况，使用GradientDesent方法，不断的去调整周围词的向量。预测次数/复杂度大概是O(V)
      - 分层Softmax：将一个V分类分解为一系列的二分类
        - 输出层是一个基于词频的哈夫曼树
        - 计算达到某个词 Wi 的路径总似然 (σ为Sigmoid函数),即为某个二分类的概率
        - ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/a20f2616-522f-49d4-8c10-81d65ceb7b47)
  
    - **Skip-Gram** + Negative Sampling    77-91
      - Skip-Gram: 用中心词来预测周围的词，从而利用周围的词的预测结果情况，使用GradientDecent来不断的调整中心词的词向量。预测次数/复杂度大概是O(KV)，假设上下文窗口为K（即目标词前、后各取K个词）
      - 负例采样分布：
        - ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/c834e930-eee0-43a8-9bde-c4f7d4f5d59a)
        - ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/2fae2784-198d-4189-83f0-0e69a99249e5)
        - 参数优化调整：误差反传、梯度下降（随机梯度下降法 SGD）
    - 评估 93-96
      - 外部任务、内部任务（词相似度、词类比） 
    - 发展 97-110 
      -GloVe（Global vector）： 利用全局统计信息， 词（目标词）-词（上下文）的同现概率的比例
      ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/fb6db2a4-3ce8-48dc-ba3f-2c4f2fd7fe37)
      ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/9f627e35-f5ef-470c-b992-4356603e1f1a)





# 句子分析

## 一、句法分析
### （1）形式语言（L6）
- 有两种不同的句法结构：依存结构、短语结构
  - 依存结构：说明词和其它词之间的依赖关系（从属关系、支配关系等）。依存关系描述为从head (被修饰的主题) 用箭头指向dependent (修饰语)
  - 短语结构：使用短语结构语法，将句子表示成嵌套的短语成分
    ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/efc39810-06c2-4f39-bb8a-7fcd68f87b35)
      
- L6中的句法分析（Parsing）：专指短语结构分析
  - 输入：句子； 输出：句法树
  - 短语：     S（sentence，句子）、NP（名词短语）、VP（动词短语）、PP（介词短语）
  - 词性类别：  N（noun，名词）、 V（verb，动词；而且 Vi为不及物动词、Vt为及物动词）、P（介词）、DT（determiner， 限定词the）

- 形式语法
  - **形式语法是一个4元组 G=(N, ∑, P, S)**
    - N 是非终结符的有限集合(有时也叫变量集或句法种类集)；
    - ∑/T 是终结符的有限集合，N ∩ ∑=∅ ；    V=N ∪ ∑ 称为总词汇表；
    - P/R 是一组重写规则的有限集合：P={α ——> β}，一个初步的字符串 α 通过不断地运用重写规则，就可以得到另一个字符串 β， 其中，α 是 V 中元素构成的串，但α中至少应含有一个非终结符号；
    - S ∈N，称为句子符或初始符。
  - 两种推导过程：最左推导、最右推导（规范推导）
    - **例题3-1**（两种方法，PPT 21-29）
  - 句型：一些特殊类型的符号串为  文法 G=(N, ∑, P, S) 的句子形式
  - 句子：文法 G 的不含非终结符的句子形式称为 G 生成的句子
  - 语言：由文法 G 生成的语言，记作 L(G)，指 G 生成的所有句子的集合
 
  - 形式语法 G 的类别（4类）：每一个正则文法都是上下文无关文法，每一个上下无关文法都是上下文有关文法，而每一个上下文有关文法都是无约束文法
    - 1、正则文法（或 3型文法）
       - 文法 G=(N, ∑, P, S) 的 P 中的所有规则满足如下形式：（其中 A、B ∈ N 非终止符 ； x ∈ ∑ 终止符）
        - 左线性正则文法：A ——> B x 或 A ——> x
        - 右线性正则文法：A ——> x B
           - **例3-2**由于（C）不满足，因此不是右线性正则文法
     - 2、上下文无关文法(context-free grammar, CFG) （或 2型文法）
       - 文法 G=(N, ∑, P, S) 的 P 中的所有规则满足如下形式：（其中 A ∈ N 非终止符 ； α ∈ (N ∪ ∑)* ）
         - A ——> α
           - **例3-3** 是上下文无关文法 
     - 3、上下文有关文法((context-sensitive grammar, CSG) （或 1型文法）
        - 文法 G=(N, ∑, P, S) 的 P 中的所有规则满足如下形式：（其中 A ∈ N 非终止符 ； α、β、γ  ∈ (N ∪ ∑)* 且 γ至少包含一个字符 ）
         - α A β ——> α γ β
           - **例3-4** 是上下文有关文法
     - 4、无约束文法（无限制重写系统）（或 0型文法）      
  - 如果一种语言能由几种文法所产生，则把这种语言称为在这几种文法中受限制最多的那种文法所产生的语言。 **例3-5**为上下文无关文法

- CFG（上下文无关文法）产生的语言句子的派生树表示
  - ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/99cad028-6b1d-49c8-b4b2-11443220d309)
  - ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/d2f846d5-b3ef-483e-b291-83651f89f16c)
- CFG（上下文无关文法）的二义性
  - 一个文法 G，如果存在某个句子有不只一棵分析树与之对应，那么称这个文法是二义的。
  - ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/707fad7c-8cd1-4a2f-b1cf-2e271dfb20fa)
![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/c3db0141-09fc-493d-98b4-30568dc7665b)
![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/a0b916e3-455f-4ebb-8721-82a157904f7d)
![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/faffb30e-5f18-4d4b-a154-384e8af086cd)
![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/86e6602c-284a-4fba-abe1-d57c332fd06f)

### （2）句法分析2（L7）
- 一个受Chomsky范式约束的CFG句法 G = (T, N, S, R)   所有的 Chomsky 范式的文法都是上下文无关，反过来，所有上下文无关文法都可以有效的变换成等价的 Chomsky 范式的文法
  文法 G=(T, N, S, R) 的 R 中的所有规则满足如下形式：（其中 A、B、C ∈ N 非终止符 ； α ∈ T 终止符）
  - A ——> BC 或 A ——> α
  - 即生成式右侧仅能出现以下两种情况：两个非终结符 / 一个终结符
- 应用句法规则生成句子、应用句法规则构建句法树
  -![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/0398c9f9-4676-4b3c-8054-d59d22488e86 width=10)
- **CKY句法分析**
  - CKY算法
    - ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/3282b854-9b92-4d34-8a3d-4c36ba842b1d)
    - ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/53b33a18-a8ff-48cf-a9f4-4507c4f5961f)
    - 例题
      - ![image](https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/70fa2c15-737b-4a83-a6d6-0d1cc707e7a4)




对于长度从1到句子长度n的每一个值： // 非终结符规则
    对于起始位置从1到n - (长度 - 1)
        终止位置 = 起始位置 + 长度 - 1
        对于中间位置从起始位置到终止位置 - 1： // 二元规则
            对于所有非终结符X在[start, middle]范围内：
                对于所有非终结符Y在[middle + 1, end]范围内：
                    对于规则A → XY：
                        在位置[start, end]添加新的图表条目A

        对于所有非终结符X在[start, end]范围内： // 一元规则
            对于规则A → X：
                在位置[start, end]添加新的图表条目A



 



  - 输入句子，输出（短语）句法树 L6, P1-13
    - 句法树
      - 短语：NP，VP
      - 词的词性类别：N，V，DT
  - 形式语言

    - L6 P14-45
      - 语法
      - 推导
      - 句型和句子
      - 上下文有关文法
    - 上下文无关语法 （CFG）L7
      - 定义 3-5
      - Chomsky 范式 P6
      - 构建句法树
        - CKY 算法 P10-29
    - 概率上下文无关语法（PCFG）L7 P35-63
  - 评价 P65-66
- 基于依存关系的句法分析（依存句法树要会画）

### （3）依存语法分析（L8）




## 二、句义分析

- SRL
- ~~RNN / Attention (不考)~~

## ~~篇章分析(不考)~~

# NLP应用

- 分类(只考最基本的基于统计的分类模型，不考基于神经网络的)
- MT(机器翻译, 看例题)
- 人机对话(考概念)
- RS(推荐系统)
- ~~文本聚类(选学，不考)~~
- ~~信息检索（选学，不考）~~
