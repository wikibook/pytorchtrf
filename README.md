# 파이토치 트랜스포머를 활용한 자연어 처리와 컴퓨터비전 심층학습

![pytorch.jpg](pytorch.png)


## Datasets

이 책에서 사용된 데이터세트는 [캐글](https://www.kaggle.com/datasets/s076923/pytorch-transformer)에서 다운로드할 수 있습니다.

> https://www.kaggle.com/datasets/s076923/pytorch-transformer

## Notice

각각의 예제는 실행 환경 및 라이브러리/프레임워크 버전에 따라 결과가 다를 수 있습니다.

이 책에서 사용된 라이브러리/프레임워크 버전은 다음과 같습니다.

```
absl-py==1.4.0
cython==3.0.2
datasets==2.14.5	
evaluate==0.4.0	
fastapi==0.103.1	
flask==2.3.3	
gensim==4.1.2
imgaug==0.4.0
jamo==0.4.1
konlpy==0.6.0
korpora==0.2.0
lightning==2.0.1
nlpaug==1.1.11
nltk==3.7
opencv-python==4.8.0.76
pandas==1.4.4
pillow==9.2.0
portalocker==2.7.0
pycocotools==2.0.7	
requests==2.28.1
sacremoses==0.0.53
scikit-learn==1.0.2	
sentencepiece==0.1.97
spacy==3.6.1
streamlit==1.22.0
tensorly==0.8.1
timm==0.9.7	
tokenizers==0.13.3	
torch==2.0.1	
torchdata==0.6.1	
torchinfo==1.8.0	
torchtext==0.15.2	
torchvision==0.15.2	
transformers==4.33.2	
ultralytics==8.0.128
```

일부 라이브러리/프레임워크는 안정적인(stable) 버전이 변경될 수 있으므로, 이 책에서 사용한 버전 설치가 불가능할 수 있습니다.

이런 경우 책에서 사용한 버전과 유사한 버전으로 설치합니다.

### MacOS M1/M2 : Device

이 책은 CUDA 가속을 기반으로 작성되어 애플 실리콘이 탑재된 맥에서는 일부 코드가 실행되지 않거나 GPU 가속이 되지 않습니다.

애플 실리콘이 탑재된 맥에서는 cuda 메서드가 지원되지 않으므로, MPS(Metal Performance Shaders) 가속을 사용합니다.

애플 실리콘 사용자는 다음과 같이 to 메서드로 MPS 가속을 적용할 수 있습니다.

```python
import torch


device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
tensor = torch.FloatTensor([1, 2, 3]).to(device)
```

### MacOS M1/M2 : Operator not implemented 

파이토치의 일부 연산은 MPS 가속이 불가능할 수 있습니다.

MPS 경고가 발생하는 경우 다음과 같이 환경 설정을 변경합니다.

```python
import os


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 설정은 MPS를 활성화하고, MPS를 지원하지 않는 환경에서도 파이토치가 작동할 수 있도록 설정합니다.

이 설정은 MPS 문제가 발생하는 환경에서 파이토치가 비활성화되지 않도록 합니다.

MPS 가속 사용 여부는 사용중인 하드웨어와 파이토치 버전에 따라 다를 수 있으므로 상황에 맞게 설정을 조정해야 합니다.


### _torchtext.so : Symbol not found

일부 라이브러리는 현재 사용 중인 파이토치 버전과 호환되지 않을 수 있습니다.

예를 들어 토치 텍스트는 파이토치 버전과 호환되어야 합니다.

이러한 경우, 다음과 같이 삭제한 후 재설치하여 버전 호환성을 맞출 수 있습니다.

```
pip uninstall torch torchtext
pip install torch torchtext
```

파이토치를 재설치할 때, CUDA 버전과의 호환성을 반드시 확인합니다.

### Multi30k : Timeout

Multi30k 데이터세트 다운로드 시 타임아웃 오류가 발생하는 경우, 데이터세트의 URL 주소를 변경해야 합니다.

URL 주소를 변경하는 방법은 다음과 같습니다.

```python
from torchtext.datasets import multi30k


multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz"
```

## Contacts

- 윤대희 : [s076923@gamil.com](mailto:s076923@gmail.com)
- 김동화 : [dhcycle25@gmail.com](mailto:dhcycle25@gmail.com)
- 송종민 : [whdwhd93@naver.com](mailto:whdwhd93@naver.com)
- 진현두 : [gusen0927@gmail.com](mailto:gusen0927@gmail.com)
