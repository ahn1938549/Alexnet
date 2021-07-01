# Word2vec

## Introduction

-   N-gram의 개념을 소개 
- 이는 정해진 길이 단위로 문장을 잘라 인덱싱하여 특정 단어를 인식하는 방법이며
- 알고리즘이 단순하고, 검색 누락이 발생하지 않기에 검색엔진에서 사용하는 경우가 많습니다.
- 하지만, 이는 검색시 노이즈가 많이 크고, 문장이 길어질수록 용량을 많이 사용하는 단점이 있습니다.
- 또한, NLP와 같은 알고리즘도 개별적인 단어만 표현할 뿐 유사성을 표현하지 못하여 시간과 용량의 지연 문제가 있다.
- 현재 머신러닝은 큰 데이터 셋을 사용하는 것이 가능하기에, 이를 사용하여 word vector를 제안합니다.

-   예시
- vector(king) - vector(man) + vector(woman) = vector(queen)

## Model
- 잠재 의미 분석(Latent Semantic Analysis) : 선형회귀를 이용하여 단어의 행렬을 생성하여 함축해 각 문헌이 어떤 잠재 의미군에 속하는 용어들을 많이 포함하고 있는지 비율을 나타내는 알고리즘.을 참고하여 개발하였다.
- ![first](https://user-images.githubusercontent.com/69898343/124156569-c0f1ee00-dad2-11eb-83f8-cb9d4f163039.png)
-   다음과 같은 훈련의 시간 복잡도를 나타낸다.
- E = epoch(대부분 3 ~ 50), T = training set의 word 갯수(more then 10억), Q = 모델별 특성
- 그 후, backpropagation과 gradient descent로 수렴도를 확인한다.
- 
### Feedforward Neural Net Language Model (NNLM)
- 기존 NNLM
- input, projection, hdiien, output layer로 구성
- 모든 단어를 one hot 벡터에 집어 넣어, 예측 레이블을 생성한다.
- ![image](https://user-images.githubusercontent.com/69898343/124157256-8a68a300-dad3-11eb-8784-843fec93c2e6.png)
- what will the fat cat sit on의 문장으로 NNLM을 만들었을 때.
- 이를 정해진 n개만큼만 계산에 사용하며, 이를 각 값마다 가중치를 곱하여, 임베딩 벡터를 생성한다.
- 이것을, linear하게 만들어 softmax함수를 이용해 결과값을 도출해낸다.
- ![image](https://user-images.githubusercontent.com/69898343/124159814-85f1b980-dad6-11eb-8dd0-066672b08be0.png)

- 단점 : n개의 갯수만 참고하도록 설정해놓으면, 나머지 단어를 버려 무의미한 단어들을 참고할 경우가 생긴다.
- 이를, Huffman binary Tree를 사용하여 복잡도를 log2(V) -> log2(Unigram perplexity(V))로 절약하였다.
- Huffman binary Tree란 출현 빈도가 높은 데이터를, binary code 중 가장 짧은 수로 나타내어, bit 수를 절감하는 방법.

다만, 이는 hidden layer를 통하여 계산하기 때문에, 복잡성이 비교적 높아 다른 알고리즘을 사용한다.

### Recurrent Neural Net Language Model (RNNLM)
- 비교적 얕은 신경망인 RNN을 사용하여 복잡한 패턴을 최소화하며, projection layer를 없앨 수 있다.
- 연산은 동등하게 softmax로 진행한다.

### Parllel Training of Neural Networks
- 큰 데이터 set을 train 하기 위하여, multi - core를 사용하여, 같은 모델에 대하여 복제본을 만들어 학습을 병렬적으로 실행합니다.
- mini - batch를 사용하며, Adagrad를 이용하여 기울기의 손실을 줄이며, 모든 업데이트는 중앙 서버에서 동기화시킵니다.
- ![image](https://user-images.githubusercontent.com/69898343/124161653-a15dc400-dad8-11eb-9e06-68a7ae514e41.png)
-   DistBelief 기법의 예시

### New Log-linear Models
- 복잡도를 더 줄이기 위하여 hidden layer를 사용하지 않고 완전히 새로운 알고리즘을 제시.
- Continuous Bag-of Words Models(CBoW)와 continuous skip-gram model(Skip-gram)
- Continuous Bag-of Words Models(CBoW) : NNLM에서 projection layer를 사용하며, hidden layer를 제거
- 모든 단어들이 같은 가중치로 projection 되는 방법. 
- NNLM은 이전 단어들만 사용하여 예측하였지만, CBoW는 이전/이후의 단어들을 사용하여 예측을 진행 
- hidden layer를 없애, 하나의 vetor로 압축하는 것이 아닌 단순한 평균을 내어 복잡도를 많이 줄였습니다.
- (hidden layer의 activation function이 연산량의 절반 이상을 차지)

- 결론 : 학습에서 과거와 미래의 단어들을 사용하여 다음 단어를 예측한다.

- continuous skip-gram model(Skip-gram) : CBow의 input과 output이 변환 알고리즘
- 같은 문장안의 다른 단어들을 전부 넣어 예측 알고리즘을 수행하기 떄문에, 연산량은 CBow에 많은 편
- 예측 하는 단어의 근처의 단어일수록, 가중치를 더 높여 중요하게 만든다.
- 평균을 내서 결론을 내기 보다는, 온전히 값 그대로를 가져오기에 train의 효과가 좋다.

- 결과 : 문장의 다른 단어들을 다 넣어, 다음의 단어를 예측
- ![image](https://user-images.githubusercontent.com/69898343/124164044-45e10580-dadb-11eb-868f-3cc7b18fff0f.png)




### Results
- 단어들의 의미없는 one-hot vector에서 벗어나, 유사성 같은 단어끼리 벡터를 갖고 묶어 성능을 향상시켰다.
- ![image](https://user-images.githubusercontent.com/69898343/124163989-31047200-dadb-11eb-9075-42ca757b7b6c.png)
- 다음과 같은 pair를 만들어 dataset으로 설정하여 훈련하였다.

- Google News Corpus의 60억개의 데이터중 3만개의 자주사용되는 단어를 사용하여, vector traing을 CBow실시 한 결과
- ![image](https://user-images.githubusercontent.com/69898343/124164337-7fb20c00-dadb-11eb-984c-1811b271d083.png)
 - 다음과 같은 결과를 얻을 수 있었으며, 벡터의 차원 혹은 훈련 데이터를 키우면 정확도가 향상되는 점을 확인 할 수 있다.

- ![image](https://user-images.githubusercontent.com/69898343/124164557-c1db4d80-dadb-11eb-8fad-bb5c1ad0898d.png)

- 기존의 NNLM의 알고리즘과 비교했을 경우, 얼마나 향상되었는지 표로 확인 할 수 있으며, skip-gram의 뛰어난 효과를 확인 할 수 있다.

### 고찰
- 기존 one-hot 벡터로 단어를 표현하는 점에서 벗어나, 단어들끼리의 결합관계를 생성하고 이에 따라 알고리즘이 자동으로 대안의 단어를 학습할 수 있다는 점이 놀라웠다.

