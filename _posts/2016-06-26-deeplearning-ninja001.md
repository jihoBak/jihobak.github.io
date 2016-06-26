---
layout: post
published: true
title: DeepLearning Ninja001
subtitle: Hello Tensorflow
date: 2016-6-26
---
# Hello Tensorflow

‘Big data’에 이어 ‘(Machine, Deep) Learning’ 라는 단어가 요즘에 엄청 뜨거운 것 같습니다. 얼마 전 Tensorflow-KR의 행사에 참가했었을때 사람들로 가득찬 구글 캠퍼스를 볼 수 있었습니다.( 8월에 곧 열릴 Pycon 세션 목록만 보더라도 머신러닝과 딥러닝 세션이 작년에 비해 엄청 많다는… ) 얼마 안되었습니다만, 저는 현재 혼자서 머신러닝을 공부하고 있습니다 (힘들어요 ㅜㅜ).

 앞으로 제가 공부하면서 익힌것들을 정리도 하고 또 혹시나 다른 분들께 도움이 될 수 있을까 하는 생각에  머신러닝과 관련해서 포스팅을 시작하고자 합니다. 오늘은 머신러닝의 간략한 소개와 앞으로의 머신러닝이라는 여행의 동반자가될 Tensorflow의 특징들을 정리 해보겠습니다. Tensorflow에 부분의 설명은 최근에 oreilly에 올라온 [Hello Tensorflow](https://www.oreilly.com/learning/hello-tensorflow) 라는 글에서 참조 하였습니다.
 
이 글에서는 Tensorflow의 특징을 살펴보고 이를 아주작은 인공뉴런을 텐서플로우로 구현해보면서 딥러닝의 간을 봅니다.

- 머신러닝?
- 이름만 알아도 반은 안다.Tensor 와 Flow를 알아보자
	- TensorFlow 특징(graph, Session)
    - TensorBoard
- TensofFlow로 작은 인공뉴런 만들어보기
 
 
 본격적인 설명으로 들어가기전 머신러닝과 한 가지 컨셉부터 분명히 가지고가는게 좋을 것 같습니다. Data-Driven Approach 입니다. 예를들어 보겠습니다. 남자와 여자의 사진을 보고 성별을 구분하는 프로그램을 만들어라 했을때, 어떤 생각이 드시나요? 남자는 머리가 짧으면 남자라고 정의할까요? 이렇게 되면 머리가 짧은 여자분은 또 머리가 긴 남자분의 사진이 주어지면 컴퓨터는 틀리게 되죠. 사과와 오렌지를 구분하는 경우에는 빨간것은 사과라고 정의 할까요? 그러면 초록색 사과는 어떻게 될까요? 이렇게 현실적으로 남자와 여자의 특징, 사과와 오렌지의의 특정을 모두 정의하긴 너무나 어렵습니다.
 

 이처럼 남자는 머리가 짧고, 키가 크며…  사과는 빨갛며...

```python
if (hair < 15cm) and (tall > 180cm):
	this is man
    ...
if color is red:
	this is red
    ...
```


 이런식으로 하나 하나 특징들에대한 미리 특정 요소들을 명시적으로 정해두고 판단하는 접근과 달리 data-driven approach는 컴퓨터에게 남자 사진을 수십만장을 보여주고 이는 남자다. 여자 사진을 수십만장을 보여주고 이것은 여자다 또는 사과와 오렌지 사진을 수십만자을 보여주고 이것은 사과다 이것은 오렌지다라고 학습을 시키는 접근 방법(without being explicitly programmed)입니다. 이때 컴퓨터가 처음 부터 능동적으로 배우지는 못하니 사람이 학습(learning) 알고리즘을 프로그래밍 해주고 이에 따라 컴퓨터는 양질의 데이터가 많이 들어 올 수록 점점 더 주어진 task를 잘 완수하는 기계(Machine)가 되어 갑니다. 이때 기계가 데이터를 학습하는 것을 말 그대로 머신러닝이라고 부릅니다. 러닝의 종류에는 수 많은 방법들이 있겠고 그 중에서 사람의 사람의 신경망에서 힌트를 얻은 학습방법을 사용하는 머신러닝을 딥러닝이라고 부릅니다. 이제 저 같은 사람들은 이 학습 알고리즘들을 수학적, 통계적 접근방법들을 통해 공부하는 것이죠.
 
 > "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E" __Tom M. Mitchell

머신러닝의 정의: 즉, 어떠한 태스크(T)에 대해 꾸준한 경험(E)을 통하여 그 T에 대한 성능(P)를 높이는 것, 이것이 기계학습이라고 할 수 있다.[[나무위키](https://namu.wiki/w/%EA%B8%B0%EA%B3%84%ED%95%99%EC%8A%B5)]

---

여기서 두 가지를 생각해 볼 수 있습니다.

1. 머신러닝이 잘 되려면 데이터가 많이 필요하다.
2. 보다 빠른 시간안에 데이터를 짧은 시간안에 머신러닝을 할려면 고성능 컴퓨팅이 필요하다.

이를 보면 왜 지금에서야 이토록 머신러닝, 빅데이터가 핫한지 알 수있죠.
- 첫째로 디지털 환경의 보편화로 어마어마한 데이터가 쌓이고 있고
- 둘째로 과거와 달리 고성능 컴퓨팅환경(cpu, gpu, cloud)을 비교적 저렴한 가격에 보다 쉽게 구축 할 수 있어서 경제성있는 컴퓨팅을 할 수 있게 되었습니다. 예를들어 컴퓨팅 스펙이 바쳐주지 않아서 1년정도 학습시간이 걸린다면... 안되겠죠.


### 이어서 바로 TensorFlow를 알아 보도록 하겠습니다.  

  
  
  
  ![tensorflow_logo](https://upload.wikimedia.org/wikipedia/en/7/74/TensorFlow.png)


 많은 데이터를 다뤄야하니 많은데이터를 효과적으로 다룰 자료구조가 필요하겠죠? TensorFlow에서는 이를 Tensor라고 부릅니다. TensorFlow에서 Tensor는 다차원의 array, list라고 생각하시면 됩니다. 간단하게 말하자면 행렬이라고 생각하시면 될 것 같습니다(You can think of a TensorFlow tensor as an n-dimensional array or list).

```python
tensor1 = 7
tensor2 = [7]
tensor3 = [[1,2,3],
		   [4,5,6]]
           ...
```

TensorFlow라는 이름에 이제 flow가 남아있습니다. 눈치 채셨겠지만 TensorFlow는 이 Tensor의 흐름(Flow)을 요리조리 쉽고 멋지게 가지고 놀 수 있게해주는 라이브러리 입니다. 이제 TensorFlow의 로고가 다르게 보이실 겁니다. Flow 는 뒤에 graph에서 한 번더 이야기 하겠습니다.

 이런 다차원 array인 Tensor들의 연산에 적합한 하드웨어는 무엇일까요? CPU 보다는 GPU(Graphics Processing Unit)입니다 훨씬 빠르게 Tensor들의 연산을 실행 할 수 있습니다. 그래서 본격적으로 딥러닝을 시작하시게되면 필수적으로 GPU가 필요하실 것 입니다. 이외에도 최근에 구글은 머신러닝을 위해 만든 TPU(Tensor Processing Unit)를 만들어버렸습니다. 구글 검색, 구글 스트릿뷰 등 많은 영역에서 이미 사용되어왔고 최근에는 이세돌과 바둑을 둔 알파고도 이 TPU를 사용했었다고 합니다. 아래사진은 TPU 입니다.

![Tensor Processing Unit](https://3.bp.blogspot.com/-Pv1QyUVlX20/Vz_iPo-qnQI/AAAAAAAACq8/mgLCTGT5M3QeM4nHZZBeiZp78GmuTWYowCLcB/s1600/tpu.png)

## 왜 TensorFlow 인가?

  
  
  
1. 구글이 만들고 사용한다.
2. Python을 통해서 사용 할 수 있다.

저 두 가지 이유로도 충분히 많은 장점들이 느껴지시죠? TensorFlow는 구글 브레인팀이 만들었습니다. deep learning 뿐만 아니라 다른 머신러닝 알고리즘에서도 잘 활용 할 수 있도록  만들어졌고, 여러 머신러닝 알고리즘들을 Python을 통해서 쉽고(?) 빠르게 만들어보고 실행시켜 볼 수 있습니다.


Python과 Tensorflow를 이어주는 일종의 연결고리는 무엇일까요? Numpy 입니다. Python을 사용하는 입장에서 어떻게 보면  Tensorflow를 DeepLearning을 위한 Numpy의 확장판이라고 볼 수도 있을 것 같습니다. Python을 이용해 scientific computing한다고 하면 바로 Numpy 패키지가 바로 생각나실 것 입니다. Numpy는 TensorFlow뿐만아니라 데이터 과학 등 Python의 scientific computing의 기본이 되는 중요한 인터페이스 역할을 하는 패키지이기 때문에 공부하시면 좋을 것 같습니다. Numpy와 관련해서 이번 TensorFlowKR에서 아주 하성주님께서 멋진 발표를 해주셨습니다. 한 번 보시길 강추!드립니다.

- [Zen of Numpy](https://www.youtube.com/watch?v=Dm2wkObQSas&feature=youtu.be), 하성주
- [Slide](https://speakerdeck.com/shurain/zen-of-numpy)


아래사진은 이번 GDG Global Summit 2016에서 [Introduction to TensorFlow](https://www.youtube.com/watch?v=SMltx_mHFsY) 세션에 나온 구글의 딥러닝 사용량 증가를 보여주는 슬라이드입니다. 링크를 걸어두었으니 딥러닝이 처음이신분은 보시길 추천드립니다.

// 세션 슬라이드


## 자 이제 TensorFlow에대해서 조금 더 들어가보겠습니다.
- Graph, node, edge
- Session

// 세션 슬라이드

 위 사진을 보시면 Python이 TensorFlow Core을 사용할 수있게 해주는 API역할을 하고 있습니다. 왜이럴까요? Deep Learning의 Deep이 괜히 Deep 이 아니겠죠? (deep —> 복잡하다)
 
 1. 딥러닝을 하기 위해서 해결해야할 task들을 수 많은 변수들과 연산들을 이용해서 코딩해야하고
 2. 많은 computiation들을 빠르게 처리하기 위해서 Python으로 코딩된 여러 task들을 Python 이외의 언어로 바꿔서 실행 해야하며
 3. 이를 여러 device(cpu, gpu…) 에서 실행이 해야하며, 이런 분산환경에서 나온 결과를 다시 합치고 나누고하는 등의 복잡한 처리가 필요한 경우가 많이 생길 것 입니다. 그렇다면 이런 상황 속에서 쉽고 효과적으로 프로그래밍 하기 위해 보통 프로그래밍 방식과는 조금 다른 부분이 필요하게 됩니다. 여기에서 TensorFlow의 특징이자 저 같은 초보자들이 햇갈려하는 포인트가 나옵니다. 코딩하는 부분과 실행 단이 독립적으로 분리되어 있습니다. 바로바로 실행이 되는 Python과 달리 tensorflow에서 코드들은 Session이라는 환경에서아래에서만 실행됩니다.
 
 어떻게 보면 TensorFlow 라이브러리를 써서 짠 코드들은 TensorFlow의 Core Execution Engine을 어떻게 사용할 것이라는 계획서와 같습니다. 이 계획서를 효과적으로 짜기 위해서 Tensorflow에서는 Graph라는 것을 사용합니다. 계획서에 담겨있는 여러 Plan들을 operation이라고 하고 이 계획서를 graph라고 볼 수 있습니다.
 
 // 내가그린 그림.
 
이 operation이 정의된 부분을 node라고 부르고 node와 node사이를 이어진 부분을 edge, 그리고 이 edge 안으로는 데이터들 즉 Tensor들이 왔다 갔다 하게 됩니다. 또, 이 operation이 다 담겨져있는 object를 graph라고 부릅니다. graph는 Tensor들을 어떻게 연산할 것인지 방법을 적은 종이가 빼곡히 들어있는 주머니와 같다고 할까요.(마치 앞에서 여러 데이터들을 Tensor로 담았던 것 처럼 이 여러 연산들을 graph라는 주머니에 담아버렸다고 저는 이해하고있습니다.) 이 graph들을 만드는 것을 위해서 말한 TensorFlow의 Flow라고 볼 수 있습니다. 마지막으로 이 graph 속 operations들은 Session이라는 공간 아래에서 한 번에 실행 됩니다. 
 

이처럼 우리가 해야할 task들을 operation 덩어리들(graph)로 적어주면 나중에 tensorflow 라이브러리가 이 operations 들을 다른 언어로 바꾸어서 다른 device에서 실행하는 등의 여러 복잡한 처리들을 실행하고 관리하는데 훨씬 편리할 것 입니다. 

건물을 짓는다면 Tensor는 건축재료, graph는 설계도, Session 은 건물이 지어지고있는 공사현장과 같다고 할 수 있습니다.

// 건설현장 사진


### 이 부분을 코드로 살펴보겠습니다.

```python
>>> graph = []
>>> operation1 = [ 'a = 1']
>>> operation1
['a = 1']
>>> operation2 = [ 'b = 2']
>>> operation3 = [ ‘c = a + b’]
>>> graph.append(operation1)
>>> graph
[['a = 1']]
>>> graph.append(operation2)
>>> graph
[['a = 1'], ['b = 2']]
```

 Python은 위 처럼 operation들이 바로바로 실행이 된다고 하면 tensorflow에서 코드들은 이렇게 바로바로 작동하지 않습니다.
 

### Tensorflow는 어떨까요

```python
>>> import tensorflow as tf
```

 이렇게 tensorflow를 import 하면 이미 벌써 내부에 _default_graph_stack에  default Graph가 생기는데요 tf.get_default_graph() 명렁어로 쉽게 접근 할 수 있습니다. 
 
```python
>>> graph = tf.get_default_graph() 
```

그러면 이 graph에 operation들이 차차 담기게 되겠죠. 현재는 비어있습니다

```python
>>> graph.get_operations()
[]
```

```python
>>> input = tf.constant(1.0)
>>> operations = graph.get_operations()
>>> operations
[<tensorflow.python.framework.ops.Operation at 0x117a440f0>]
```

 바로 볼 수는 없지만 operations에는 operation들이 들어있다는 것을 충분히 알 수 있습니다. 또 operation그안에는 operation에는 사용할 node가 들어있겠죠. 그러면 operation을 직접 확인 해보겠습니다. 
 
```
>>> operations[0].node_def
name: "Const"
op: "Const"
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
      }
      float_val: 1.0
    }
  }
}
```

무슨 JSON 비슷한게 적혀져 나오는데 이것이 바로 나중에 Session 아래에서 실행될때 tensorflow가 해석할 부분이자 graph라는 계획서에 담겨 나중에 처리될 계획들의 일 부분입니다. 이 JSON과 비슷하게 생긴것은 [Protocol buffer](https://developers.google.com/protocol-buffers/)(구글이 버전의 JSON?)
라고 합니다. Tensorflow는 내부적으로 이 Protocol buffer을 사용하구요.   

자 이제 input을 출력해보겠습니다.(Session 아래에서만 실행됩니다)
   
```python
>>>input
<tf.Tensor 'Const:0' shape=() dtype=float32>
>>> sess = tf.Session()
>>> sess.run(input)
1.0
```
단순 python 의 출력 코드가아니라 tensorflow의 graph가 실행 된 것 입니다. R계의 초고수 Hadley Wickham의 말을 가져와보겠습니다. 아래 그림이 조금 이해가 되시나요? Tensorflow를 사용하실 수록 점점 이 그림에대한 이해도가 높아지실 것 입니다.

// 해들리 위컴 사진



## '초간단' Tensorflow 뉴런 만들어보기


일단 잠깐 뉴런에대해서 살펴보겠습니다. 코딩하는데 뭐 계속 뉴런, 뉴런 할까요? 처음에 설명한 머신러닝을 보면 결국 머신러닝은 데이터를 입력받아 학습해서 원하는 출력값을 만들어내는 함수로 볼 수있는데요. 뉴런도 가만히 살펴보니 단순화 시키면 일종의 input을 받아서 output을 출력하는 함수(Function)로 볼 수 있다는 것입니다. 그런데 실제 머신러닝에서 필요한 입력과 함수가 한두개의 일까요? 

//심플사진

 아닙니다. 수많은 입력값들을 수많은 함수들을 통해서 처리해야하는데요, 뉴런이 함수와 비슷하니 이 함수들을 우리 몸의 신경망처럼 들을 복잡하게 연결시켜놓으면 어떨까? 하고 만들어넨 머신러닝이 딥러닝입니다. 

// 복잡사진   


  그러나 여기서 tensorflow 코드로 만들어볼 뉴런은 입력이 하나 출력이 하나인 초간단 tensoflow 뉴런입니다. 
  
  
