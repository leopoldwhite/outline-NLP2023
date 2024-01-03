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
    - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/4a765052-eb79-43fe-8507-6e388d3ef144" width="350" height="200">
    - 降维 50-51
      - 选择某些维（如TF-IDF）、SVD进行分解、PCA降维
  - 2、基于预测的低维向量
    - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/99c2fdb8-ba0c-46af-8f52-800d5459cfe4" width="500" height="200">
    - **CBOW** + Hierarchical SoftMax    59-75
      - CBOW：用周围词预测中心词，从而利用中心词的预测结果情况，使用GradientDesent方法，不断的去调整周围词的向量。预测次数/复杂度大概是O(V)
      - 分层Softmax：将一个V分类分解为一系列的二分类
        - 输出层是一个基于词频的哈夫曼树
        - 计算达到某个词 Wi 的路径总似然 (σ为Sigmoid函数),即为某个二分类的概率
        - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/a20f2616-522f-49d4-8c10-81d65ceb7b47" width="350" height="200">
  
    - **Skip-Gram** + Negative Sampling    77-91
      - Skip-Gram: 用中心词来预测周围的词，从而利用周围的词的预测结果情况，使用GradientDecent来不断的调整中心词的词向量。预测次数/复杂度大概是O(KV)，假设上下文窗口为K（即目标词前、后各取K个词）
      - 负例采样分布：
        - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/c834e930-eee0-43a8-9bde-c4f7d4f5d59a" width="350" height="100">
        - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/2fae2784-198d-4189-83f0-0e69a99249e5" width="250" height="100">
        - 参数优化调整：误差反传、梯度下降（随机梯度下降法 SGD）
    - 评估 93-96
      - 外部任务、内部任务（词相似度、词类比） 
    - 发展 97-110 
      -GloVe（Global vector）： 利用全局统计信息， 词（目标词）-词（上下文）的同现概率的比例
      - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/fb6db2a4-3ce8-48dc-ba3f-2c4f2fd7fe37" width="500" height="130">
      - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/9f627e35-f5ef-470c-b992-4356603e1f1a" width="450" height="130">





# 句子分析

## 一、句法分析
### （1）形式语言（L6）
- 有两种不同的句法结构：依存结构、短语结构
  - 依存结构：说明词和其它词之间的依赖关系（从属关系、支配关系等）。依存关系描述为从head (被修饰的主题) 用箭头指向dependent (修饰语)
  - 短语结构：使用短语结构语法，将句子表示成嵌套的短语成分
    - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/efc39810-06c2-4f39-bb8a-7fcd68f87b35" width="500" height="130">
      
- L6中的句法分析（Parsing）：专指**短语结构**分析
  - 输入：句子； 输出：句法树
  - 短语：     S（sentence，句子）、NP（名词短语）、VP（动词短语）、PP（介词短语）
  - 词性类别：  N（noun，名词）、 V（verb，动词；而且 Vi为不及物动词、Vt为及物动词）、Adj（形容词）、Adv（副词）、P（介词）、DT（determiner， 限定词the）

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
  - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/99cad028-6b1d-49c8-b4b2-11443220d309" width="450" height="150">
  - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/d2f846d5-b3ef-483e-b291-83651f89f16c" width="350" height="200">
- CFG（上下文无关文法）的二义性
  - 一个文法 G，如果存在某个句子有不只一棵分析树与之对应，那么称这个文法是二义的。
  - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/707fad7c-8cd1-4a2f-b1cf-2e271dfb20fa" width="350" height="200">
- <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/c3db0141-09fc-493d-98b4-30568dc7665b" width="250" height="130">
- <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/a0b916e3-455f-4ebb-8721-82a157904f7d" width="250" height="130">
- <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/faffb30e-5f18-4d4b-a154-384e8af086cd" width="250" height="130">
- <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/86e6602c-284a-4fba-abe1-d57c332fd06f" width="250" height="130">

### （2）句法分析2（L7）
- **一个受Chomsky范式约束的CFG句法** G = (T, N, S, R)   所有的 Chomsky 范式的文法都是上下文无关，反过来，所有上下文无关文法都可以有效的变换成等价的 Chomsky 范式的文法
  文法 G=(T, N, S, R) 的 R 中的所有规则满足如下形式：（其中 A、B、C ∈ N 非终止符 ； α ∈ T 终止符）
  - A ——> BC 或 A ——> α
  - 即生成式右侧仅能出现以下两种情况：两个非终结符 / 一个终结符
- 应用句法规则生成句子、应用句法规则构建句法树
  -<img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/0398c9f9-4676-4b3c-8054-d59d22488e86" width="400" height="200">
- **CKY句法分析**
  - CKY算法  P10-29
    - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/3282b854-9b92-4d34-8a3d-4c36ba842b1d" width="350" height="200">
    - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/53b33a18-a8ff-48cf-a9f4-4507c4f5961f" width="350" height="300">
    - 例题
      - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/70fa2c15-737b-4a83-a6d6-0d1cc707e7a4" width="350" height="230">
- 句法分析可能造成的歧义
  - 词性歧义、名词修饰语歧义、介词短语修饰语歧义、边界歧义
- **概率上下文无关法**（ Probabilistic context-free grammars (PCFGs）  或者   Stochastic context-free grammars (SCFGs) ）   P35-63
  - 一个概率上下文无关文法可以表示为一个五元组 G = (T, N, S, R, P)
    - 其中，P：概率函数，为每个重写规则赋予一个概率值
    - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/92534ae4-936f-40b7-a120-ecbbde2e2d3d" width="500" height="100">
  - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/831e5468-a9a7-42ca-9b3e-fb0b1a8eccb2" width="500" height="100">
    - 对于句子s和其可能的句法导出树集合Γ(s)，PCFG为Γ(s)中的每棵树t赋予一个概率 p(t)，即得到候选树按照概率的排序。概率最大对应的树 即为句子S最有可能的句法树
  - PCFG的两个问题：句法规则学习得到PCFG、基于PCFG的句法分析从从多个候选树中找出一个概率最大的树
    - 从treebank（词库）中统计重写规则、并计算其概率
  - PCFG的三个假设：位置不变性、上下文无关、祖先节点无关
  - PCFG中的向外、向内概率
  - PCFGs的三个基本问题：● 计算句子的概率：P(w1 m|G)    ● 为句子找到最优句法树：argmaxt P(t|w1 m;G)     ● 参数学习：求解使得P(w1 m|G) 最大的句法G
    - 动态规划表求解问题2  ：π(i, j, X)
- 评价 P65-66

### （3）依存语法分析（L8）      基于依存关系的句法分析（依存句法树要会画）
- 现代依存语法(dependency grammar)理论
  - 结构句法可概括为关联、组合、转位这三大核心。句法关联建立起词与词之间的从属关系，这种**从属关系**是由支配词和从属词联结而成；动词是句子的中心，并支配其他成分，它本身不受其他任何成分的支配。
  - “依存”就是指词与词之间支配与被支配的关系，这种关系不是对等的，而是有方向的。处于支配地位的成分称为**支配者**(governor,regent, head)，而处于被支配地位的成分称为**从属者**(modifier, subordinate, dependency)。
  - 动词的价数：该动词所能支配的行动元(名词词组)的个数。也就是说，它能支配几个行动元，它就是几价动词。
  - 有向图中，用带有方向的弧(或称边，edge)来表示两个成分之间的依存关系，支配者在有向弧的发出端，被支配者在箭头端，我们通常说被支配者依存于支配者。
  - 依存语法的4条公理
    - (1) 一个句子只有一个独立的成分；(2) 句子的其他成分都从属于某一成分；(3) 任何一成分都不能依存于两个或多个成分；(4) 如果成分A直接从属于成分B，而成分C在句子中位于A和B之间，那么，成分C或者从属于A，或者从属于B，或者从属于A和B之间的某一成分。
    - 这4条公理相当于对依存图和依存树的形式约束为：单一父结点(single headed)、连通(connective)、无环(acyclic)、可投射(projective)，因此句子的依存分析结果是一棵有“根(root) ”的树结构。
    - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/7ea874c7-5be1-4a5c-8fb6-d769fc07ff2c" width="350" height="200">
- 依存语法分析方法
  - 建立一个依存句法分析器一般需要完成以下三部分工作： (1) 依存句法结构描述  (2) 分析算法设计与实现  (3) 文法规则或参数学习
  - 依存句法结构描述：一般采用有向图或依存树，所采用的句法分析算法可大致归为以下4类：生成式的分析方法、判别式的分析方法、决策式的(确定性的)分析方法、基于约束满足的分析方法。
    - （1）生成式的分析方法
      - 基本思想：采用联合概率模型Score(x, y|θ)(其中，x 为输入句子，y 为依存分析结构，θ 为模型的参数)生成一系列依存句法树，并赋予其概率分值，然后采用相关算法找到概率打分最高的分析结果作为最后输出。
        - 模型A. 二元词汇亲和模型
          - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/d4c3a364-570f-4fc3-a252-cb501944d5bb" width="600" height="100">
        - 模型B. 选择偏好模型
          - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/89de9380-baf9-486c-af43-fd47ef999fbe" width="650" height="100">
        - 模型C. 递归生成模型
          - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/7671ff61-79d5-498f-a649-2f85fd995613" width="850" height="100">
    - （2）判别式的分析方法          
      - 基本思想：采用条件概率模型 Score(x|y, θ)，使目标函数 ∏ Score(xi | yi , θ)最大的 θ 作为模型的参数
        - 例如：最大生成树模型(maximum spanning trees, MST)定义整棵句法树的打分是树中各条边打分的加权和。
    - （3）决策式的(确定性的)分析方法
      - 基本思想：模仿人的认知过程，按照特定方向每次读入一个词。每读入一个词，都要根据当前状态做出决策(比如判断是否与前一个词发生依存关系)。一旦决策做出，将不再改变。所做决策即“采取什么样的分析动作(action)”。分析过程可以看作是一步一步地作用于输入句子之上的分析动作(action)的序列。
        - 1、移进－归约算法：使用Left-Reduce、Right- Reduce 和 Shift  这三种分析动作
          - 当前分析状态的格局是一个三元组：(S, I, A)，S, I, A分别表示栈、未处理结点序列和依存弧集合。
        - 2、Arc-eager 分析算法：使用Left-Arcl、Right-Arcl、Shift、Reduce  这四种分析动作(Actions)
          - **例题**（P 29-35）
            - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/2ad1342a-a809-499a-ab96-b20647f47fc4" width="350" height="200">
    - （4）基于约束满足的分析方法
      - 基本思想：将依存句法分析过程看作可以用约束满足问题(Constraint satisfaction problem, CSP)来描述的有限构造问题(finite configuration problem)。它不像上下文无关文法那样探索性地生成，而是根据已规定好的约束进行剪裁，把不符合约束的分析去掉，直到留下一棵合法的依存树。  
- 根据 Arc-eager 算法实现一个基于转换的依存句法分析器。
  - 基本思路：在每一个状态(configuration)下根据当前状态提取特征, 然后通过分类决定下一步应该采取的“动作”(action)：移进(shift)、左弧(left-arc)、右弧(right-arc)、归约(reduce)，执行分类器选择的最优动作，转换到下一个状态。
  - 具体实现：标注大量的依存关系句法树，建立训练集。每个句子都可以一对一地转换为动作序列；确定特征集合，以构造动作分类器。
  - **例题**：P 39-42
- 依存句法分析器评价指标
  - 1、无标记依存正确率(unlabeled attachment score, UA)：所有词中找到其正确支配词的词所占的百分比，没有找到支配词的词(即根结点)也算在内。
  - 2、带标记依存正确率(labeled attachment score, LA)：所有词中找到其正确支配词并且依存关系类型也标注正确的词所占的百分比，根结点也算在内。
  - 3、依存正确率(dependency accuracy, DA)：所有非根结点词中找到其正确支配词的词所占的百分比。
  - 4、根正确率(root accuracy, RA)：有两种定义方式：(1)正确根结点的个数与句子个数的比值；(2)另一种是所有句子中找到正确根结点的句子所占百分比。对单根结点语言或句子来说，二者是等价的。
  - 5、完全匹配率(complete match, CM)：所有句子中无标记依存结构完全正确的句子所占百分比。
    - **例题**：P 45-47
    - 依存句法树常见的依存关系标签：root（根节点）、sbj（主语）、obj（宾语）、prep（介词）、nmod（名词修饰语）、nsubj（名词主语）、amod（形容词修饰语）
- 短语结构与依存结构的关系
  - 短语结构可转换为依存结构。实现方法：(1) 定义中心词抽取规则，产生中心词表;(2) 根据中心词表，为句法树中每个节点选择中心子节点;(3) 将非中心子节点的中心词依存到中心子节点的中心词上，得到相应的依存结构。
  - **例题**：P 50-54
    - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/6d1ae8c7-9e96-4b45-9ba3-35e9ec22074e" width="500" height="300">
- 汉英句法结构特点对比（P 55-66）
  - <img src="https://github.com/leopoldwhite/outline-NLP2023/assets/128705197/535ab2f8-ddbd-442f-b3fd-db0fd508482e" width="500" height="150">
    
 
## 二、句义分析
- 句子语义的不同角度：形式语言、概念语义、情感语义
- 句子语义的表示与分析（有2种：基于符号、基于数值）
一、基于符号的方法
    - 1、基于符号的句子语义表示
      - ■基于AI知识表示
      - ■基于语言学语义表示
    - 2、基于符号的句子语义分析
      - ■句法驱动的句义分析（P 10-20）
       - <句法分析>中的句法规则包含了<语义规则>、句法具有组合性
       - （句法结构：句法规则表示）——>（语义结构：谓词逻辑表示）
      - ■基于句法结构的句义分析（P 21-24）
        - 依据依存语法得到句法结构——>转化为一系列三元组——>映射到逻辑形式
      - ■基于语义语法的语义分析（P 25-27）
        - 应用于某些特定领域的特定应用，专门开发，难以推广
      - ■语义驱动的句法分析（P 28-31）
        - 语义信息以“模式-行为”的规则形式存储在字典中，模式匹配
    - 3、**语义角色标注**(语言学语义表示+语义语法驱动，Semantic Role Labeling, **SRL** ） P 32-52
      - 格语法：深层结构表现为中心动词与一组名词短语；这些名词短语与动词间存在语义关系，被称为“深层格” 。
        - 中心动词——>谓词、名词短语——>谓词的论元
        - 九个格分别为施事格、感受格、对象格、工具格、来源格、目的格、场所格、时间格、路径格。
        - 语义角色：论元在谓词所描述的事件中所扮演角色的抽象模型
          - 优势：不同的表达可能有相同的语义结构，可以作为句子的共性浅层语义表示
          - 语义角色抽象模型的具体实施方式
            - 论旨角色(Thematic roles)
              - 为所有事件定义一个有限的角色列表
            - 论旨角色的变种
              - 更具体：  为每一个动词定义一个语义角色列表，如：PropBank
                - PropBank(Proposition Bank)：标注了<动词>的语义角色的句子资源，每个动词的每一个义项有一个角色集合，但是只给出编号。
                - NomBank(Nominalization Bank)：专门用于标注<名词化结构>（nominalizations）的语义信息，为名词性成分提供语义角色标注。
                - FrameNet：每个语义场景（frame）都包含一组语义角色，这些角色描述了该场景中词语的语义用法。
              - 更一般：  提供一些高层角色proto-agent/proto-patient，如：PropBank
              - 折衷：    逐框架地定义语义角色，如：FrameNet具有相同格框架
          - 以动词为核心的句子表示、论旨角色是动词对其论元的要求
          - 格框架：该动词需要的论旨角色组合
            - 一个动词可以有多个格框架
            - 基于格框架的动词分类：具有类似的格框架的动词归为一类
        - SRL(Semantic role labeling，语义角色标注)：自动发现动词论元的语义角色
          - 角色集：FrameNet and PropBank或其他
          - 标注方法：分类、序列标注
            - 1、分类方法
              - 1)对句子进行句法分析，得到谓词及其论元
              - 2)对每个论元进行分类，指派语义角色
              - 问题
                - 1)依赖句法分析的结果
                - 2)论元分类性能依赖设计的特征： predicate、phrase type、headword、…
                - 3)论元各自独立分类，可能存在冲突： 每个分类取多个类标后再进行全局优化
            - 2、序列标注方法
              - 标注集：{B-ARG0、I-ARG₀…}
              - 标注模型：HMM\CRF\LSTM\…
              - 问题：分析越深入，标注数据越困难
          - 语义角色的应用
            - 句子语义的等价性判定
            - 句子生成：复述生成
            - 句子推理以回答问题
            - 信息抽取：事件挖掘…      
二、基于数值的方法
- 句子语义表示和语义分析的融合、直接基于词表示构建（在词向量学习模型中加入句子向量表示，如 Doc2vec：Paragraph2vec）、基于循环神经网络、基于递归神经网络、基于注意力网络
- ~~神经网络相关知识~~（不考）
  - ~~RNN / Attention~~ (不考)





# NLP的4个应用
## 一、文本分类
## 二、机器翻译
## 三、人机对话
## 四、推荐系统

- 分类(只考最基本的基于统计的分类模型，不考基于神经网络的)
- MT(机器翻译, 看例题)
- 人机对话(考概念)
- RS(推荐系统)
- ~~文本聚类(选学，不考)~~
- ~~信息检索（选学，不考）~~


## ~~篇章分析(不考)~~


