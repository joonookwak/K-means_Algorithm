# Computer Vision Algorithm: K-means Clustering for Handwritten Digit Images

본 저장소는 고려대학교 컴퓨터 비전[김창수] 과제 #4인 'K-means Clustering for Handwritten Digit Images'의 구현 코드와 관련 데이터셋을 포함하고 있습니다. 
이 과제의 목표는 MNIST 데이터셋을 사용하여 비지도 학습 알고리즘인 K-means clustering을 구현하고, 이미지 특징 표현이 클러스터링에 미치는 영향을 이해하는 것입니다.

## 📂 파일 구성 (Provided Components)

제공된 파일들은 다음과 같은 역할을 수행합니다:

| 파일명 | 설명 |
| :--- | :--- |
| `evaluate.py` | 테스트 이미지에 대한 특징 추출, 클러스터 매핑, 에러율 계산을 수행하는 핵심 모듈 |
| `features.py` | 원본 이미지를 정규화된 196차원(downsampled) 또는 784차원(raw) 특징 벡터로 변환하는 함수 포함 |
| `kmeans.py` | K-means 훈련 로직(Task A) 및 클러스터-레이블 매핑(Task B) 구현부 |
| `load_digits.py` | MNIST 바이너리 파일을 읽기 위한 함수 제공 |
| `main.py` | K-means 훈련 및 테스트 함수를 호출하고 결과를 출력하는 실행 스크립트 |
| `visualize_centroids.py` | 학습된 클러스터 중심을 이미지(`centroids.png`)로 저장하는 유틸리티 |
| `visualize_digit.py` | Matplotlib을 사용하여 개별 이미지를 시각화하는 유틸리티 |

*참고: 데이터셋 파일(`t10k-images-idx3-ubyte` 등)은 MNIST 데이터를 포함하고 있습니다.*

## 🛠 주요 구현 과제

본 과제는 다음 4가지 세부 과제로 구성되어 있습니다:

1.  **Task A: K-means Training (`kmeans.py`)**: 10개의 클러스터 중심을 초기화하고, 반복적인 Assignment 단계와 Update 단계를 통해 학습을 수행합니다.
2.  **Task B: Cluster-to-Label Mapping (`kmeans.py`)**: 훈련이 완료된 후, 각 클러스터에 가장 빈번하게 등장하는 레이블을 할당합니다.
3.  **Task C: Prediction and Evaluation (`evaluate.py`)**: 테스트 세트를 사용하여 에러율(Error Rate)을 계산합니다.
4.  **Task D: Visualize Cluster Centers**: 학습된 클러스터 중심을 시각화하고 `centroids.png`로 저장합니다.

## 🚀 실행 방법

구현이 완료된 후, 아래 명령어를 통해 과제를 실행할 수 있습니다:

```bash
python main.py
