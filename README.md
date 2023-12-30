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

## 语言模型

- n-gram
  - 建模
- 神经网络语言模型(NNLM):
  - CBOT
  - skip-gram (SG)

## 词

### 词法分析（L4）

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

### 词性标注

- HMM
- ~~CRF(不考)~~
- 序列标注

### 词义分析

- 基于符号的词义分析
- 基于数值的词义分析

## 句子分析

### 句法分析

- 基于短语结构的句法分析
  - 形式语言
- 基于依存关系的句法分析（依存句法树要会画）

### 句义分析

- SRL
- ~~RNN / Attention (不考)~~

### ~~篇章分析(不考)~~

## NLP应用

- 分类(只考最基本的基于统计的分类模型，不考基于神经网络的)
- MT(机器翻译, 看例题)
- 人机对话(考概念)
- RS(推荐系统)
- ~~文本聚类(选学，不考)~~
- ~~信息检索（选学，不考）~~
