# Text-Segmentation
Text Segmentation 관련 논문 정리

## Text Segmentation
|title|summary|
|-----|-------|
|TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages (1997)](https://www.aclweb.org/anthology/J97-1003.pdf)| |
|[A HIDDEN MARKOV MODEL APPROACH TO TEXT SEGMENTATION AND EVENT TRACKING(1998)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=674435)| |
|[Statistical Models for Text Segmentation(1999)](http://www.cs.cmu.edu/~aberger/pdf/ml.pdf)| |
|[Advances in Domain Independent Linear Text Segmentation(2000)](https://www.aclweb.org/anthology/A00-2004/)|- C99 알고리즘|
|[Latent Semantic Analysis for Text Segmentation(2001)](https://www.aclweb.org/anthology/W01-0514/)|- LSA 사용|
|[A Statistical Model for Domain-Independent Text Segmentation(2001)](https://www.aclweb.org/anthology/P01-1064.pdf)| |
|[Minimum Cut Model for Spoken Lecture Segmentation(2006)](https://www.aclweb.org/anthology/P06-1004.pdf)| |
|[Bayesian Unsupervised Topic Segmentation(2008)](https://www.aclweb.org/anthology/D08-1035/)| |
|[Hierarchical Text Segmentation from Multi-Scale Lexical Cohesion(2009)](https://www.aclweb.org/anthology/N09-1040/)| |
|[Linear Text Segmentation using Affinity Propagation(2001)](https://www.aclweb.org/anthology/D11-1026/)| |
|[TopicTiling: A Text Segmentation Algorithm based on LDA(2012)](https://www.aclweb.org/anthology/W12-3307.pdf)| |
|[Domain-Independent Unsupervised Text Segmentation for Data Management(2014)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7022635)| |
|[Text Segmentation based on Semantic Word Embeddings(2015)](https://arxiv.org/abs/1503.05543)| |
|[Unsupervised Text Segmentation Using Semantic Relatedness Graphs(2016)](https://www.aclweb.org/anthology/S16-2016/)| |
|[합성곱 신경망을 이용한 On-Line 주제 분리(2016)](http://koreascience.or.kr/article/JAKO201608965832494.page)| |
|[Text Segmentation as a Supervised Learning Task(2018)](https://arxiv.org/abs/1803.09337)|- text segmentation 위한 wiki dataset 만듦<br/>- 기존에 unsupervised, probalistic하게 해결하던 task를 supervised하게 해결 |
|[Attention-based Neural Text Segmentation(2018)](https://arxiv.org/pdf/1808.09935.pdf)| |
|[Scientific Literature Summarization Using Document Structure and Hierarchical Attention Model(2019)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8937799)| |
|[SECTOR: A Neural Model for Coherent Topic Segmentation and Classification(2019)](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00261)| |
|[LANGUAGE MODEL PRE-TRAINING FOR HIERARCHICAL DOCUMENT REPRESENTATIONS(2019)](https://arxiv.org/pdf/1901.09128.pdf)|- text segmentation으로 실험 진행|
|[BeamSeg: A Joint Model for Multi-Document Segmentation and Topic Identification(2019)](https://www.aclweb.org/anthology/K19-1054/)| |
|[BTS: 한국어 BERT를 사용한 텍스트 세그멘테이션(2019)](http://www.dbpia.co.kr.openlink.sookmyung.ac.kr:8080/journal/articleDetail?nodeId=NODE09301605)| |
|[Context-Aware Latent Dirichlet Allocation for Topic Segmentation(2020)](https://www.semanticscholar.org/paper/Context-Aware-Latent-Dirichlet-Allocation-for-Topic-Li-Matsukawa/265160ebbe56d2226bc8180330892afa5bb7c535)| |
|[Chapter Captor: Text Segmentation in Novels(2020)](https://arxiv.org/abs/2011.04163#:~:text=Books%20are%20typically%20segmented%20into,task%20of%20segmenting%20long%20texts.)| 1.  구텐버그 프로젝트에 포함된 소설을 이용해 text segmentation 데이터셋 구축 <br/>2. Local Method:<br/>* Weighted Overlap Cut(WOC): unsupervised, 각 챕터 내 빈번히 등장하는 단어가 다를것이라는 점에서  착안, 두 문장을 비교해 단어의 밀집도(overlap하는 경우)가 최소화 되는 곳을 Break point로 둠<br/>* BERT for Break Prediction (BBP): supervised, 두 문장을 비교해 두 문장이 연속적인지(같은  챕터인지) 아니면 연속적이지 않은지(break point)를 분류 문제로 계산<br/>3. Global Method using Optimization: segment의 길이를 일정하게 만드는 것이 좋은  segmentation 결과를 보여줌<br/>* 동적 프로그래밍 기법을 사용해 recursive하게 해결 |
|[Books of Hours: the First Liturgical Corpus for Text Segmentation(2020)](https://www.aclweb.org/anthology/2020.lrec-1.97.pdf)|  |
|[A Joint Model for Document Segmentation and Segment Labeling(2020)](https://www.aclweb.org/anthology/2020.acl-main.29/)| |
|[Discourse as a Function of Event: Profiling Discourse Structure in News Articles around the Main Event(2020)](https://www.aclweb.org/anthology/2020.acl-main.478/)| |
|Improving BERT with Focal Loss for Paragraph Segmentation of Novels(2020)| |
|[Topical Change Detection in Documents via Embeddings of Long Sequences(2020)](https://arxiv.org/pdf/2012.03619.pdf)| |
|[Text Segmentation by Cross Segment Attention(2020)](https://arxiv.org/abs/2004.14535)| |


## Topic Modeling
|title|summary|
|-----|-------|
|[Latent Dirichlet Allocation(2002)](https://ai.stanford.edu/~ang/papers/jair03-lda.pdf)|LDA가 처음 소개된 논문|
|[A Hybrid Neural Network-Latent Topic Model(2012)](http://proceedings.mlr.press/v22/wan12.html)| |
|[Modelling Sequential Text with an Adaptive Topic Model(2012)](https://www.aclweb.org/anthology/D12-1049/)| |
|[Learning from LDA using Deep Neural Networks(2015)](https://arxiv.org/abs/1508.01011)| |
|[Mixing Dirichlet Topic Models andWord Embeddings to Make lda2vec(2016)](https://arxiv.org/abs/1605.02019)| |
|[Contextual-LDA: A Context Coherent Latent Topic Model for Mining Large Corpora(2016)](https://ieeexplore.ieee.org/document/7545061)| |
|[Recurrent Attentional Topic Model(2017)](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14400)| |
|[Discovering Discrete Latent Topics with Neural Variational Inference(2017)](https://arxiv.org/abs/1706.00359)| |
|[A Detailed Survey on Topic Modeling for Document and Short Text Data(2019)](https://arxiv.org/pdf/1904.07695.pdf)| |
|[감정 딥러닝 필터를 활용한 토픽 모델링 방법론(2019)](http://koreascience.or.kr/article/JAKO201911263062209.page)| |

## Applications
|title|summary|
|-----|-------|
|[A Two-Stage Transformer-Based Approach for Variable-Length Abstractive Summarization(2020)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9132692)| |
