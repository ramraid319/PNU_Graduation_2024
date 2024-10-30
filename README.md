# 강화학습기반 교차로 동적 신호제어시스템


### 1. 프로젝트 소개

본 프로젝트는 강화학습을 통해 교차로의 신호를 최적화하는 프로젝트이다.

#### 1.1. 배경 및 필요성

국내에서 가장 일반적으로 사용하는 시스템은 "시간제어식 신호"이다. 이는 특정 상황별로 주기를 미리 지정해두고, 해당 상황이 발생하면 인위적인 조작을 통해 적용하는 방식이다. 하지만 기본적으로 정해진 고정된 여러 옵션에서 단순히 선택을 하는 방식이기 때문에, 정밀한 최적화가 힘들고, 인위적인 제어가 필요하기 때문에 실시간으로 대응하기에는 한계가 있다.

즉, 현재의 시스템에는 아래와 같은 문제가 있다.
- 교통 상황을 반영하지 못하는 불합리한 차량대기 시간
- 고정된 순서로만 진행되는 신호 체계
- 현행 신호 체계의 효율성과 유연성 부족


#### 1.2. 목표 및 주요 내용

본 프로젝트에서는 강화학습(DQN)을 사용하여 교차로 신호체계를 최적화한다.

- 불필요한 대기시간을 최소화하는 합리적 신호체계 구현
- 동적인 신호순서를 적용하여 효율성 증대
- 여러 교차로의 교통상황을 고려한 실시간 최적화



### 2. 상세설계

#### 2.1. 시스템 구성도

<img width="1016" alt="Tech Stack Diagram (Copy) (7)" src="https://github.com/user-attachments/assets/52787a84-3b3d-475c-b7ef-061ca538d35c">

#### 2.1. 사용 기술

- 교차로 구현 시뮬레이터 : CARLA, SUMO Simulator
- 객체 탐지 모델 : YOLOv8
- 객체 추적 알고리즘 : ByteTrack(https://github.com/ifzhang/ByteTrack)
- Python
- PyTorch

### 4. 소개 및 시연 영상

프로젝트 소개나 시연 영상을 넣으세요.

### 3. 설치 및 사용 방법

#### Carla Simulator 설치

**시스템 요구사항**

* x64 시스템
* 165GB 디스크 공간
* 최소 6GB 이상의 GPU
* 2000, 2001 TCP 포트

**소프트웨어 요구사항**

* CMake: 빌드 파일을 생성하기 위해 필요. version >= 3.15
* Git: Carla repositories를 관리하기 위해 필요.
* Make: 실행 파일을 생성하기 위해 필요. version == 3.81
* 7Zip: asset 파일의 압축 해체에 필요.
* Python3 x64: 다른 버전이나 x32 버전은 제거 권장
* [Windows 8.1 SDK](https://developer.microsoft.com/ko-kr/windows/downloads/sdk-archive/)

**Visual Studio 2019**
* C++를 사용한 데스크톱 개발
  * MSVC v140 -VS 2015 C++ 빌드 도구(v14.00)
* C++를 사용한 게임 개발
* x64 Visual C++ Toolset
* .Net framework 4.6.2 개발 도구

**Python Dependencies**
```
pip3 -V
pip3 install --upgrade pip
pip3 install --user setuptools
pip3 install --user wheel
```

**Unreal Engine**

먼저, 언리얼 엔진을 포크하여 다운로드하기 위해서는 Github 계정이 언리얼 엔진과 연동되어 있어야 합니다. 자세한 내용은 아래 링크를 확인해주세요.
[Accessing Unreal Engine Source](https://www.unrealengine.com/en-US/ue-on-github)

1. 언리얼 엔진 소스코드 클론
```
git clone --depth 1 -b carla https://github.com/CarlaUnreal/UnrealEngine.git .
```

언리얼 엔진의 경로는 C:\\\\로 하는것을 권장합니다.

2. Configuration Script 실행
```
Setup.bat
GenerateProjectFiles.bat
```

3. 개조된 엔진 컴파일
   - Visual Studio 2019로 UE4.sln 열기
   - "Development Editor". "Win64", "UnrealBuildTool" 옵션을 선택
   - Solution Explorer에서 UE4를 우클릭 후 빌드

4. 컴파일이 성공적으로 되었는지 확인하려면 Engine\Binaries\Win64\UE4Editor.exe 을 실행

**Carla 빌드**

1. Carla repository 클론
```
git clone https://github.com/carla-simulator/carla
```

2. 최신 버전의 Assets 다운로드

아래 Script 실행
```
Update.bat
```

3. Unreal Engine 환경 변수 설정

- 윈도우 시스템 환경 변수에 Unreal Engine 루트 경로를 **UE4_ROOT** 이름으로 새로 만든다.

4. Carla 빌드
**x64 Native Tools Command Prompt for VS 2019**를 실행한다.

**Python API 클라이언트 컴파일**
Carla의 루트 경로에서 아래 명령어를 실행한다.

```
make PythonAPI
```

PythonAPI/carla/dist 폴더에 .egg 파일과 .whl 파일이 생성되었는지 확인한다.

**서버 컴파일**

Carla의 루트 경로에서 아래 명령어를 실행한다.
```
make launch
```

Unreal Engine 4 Editor가 실행되면 성공적으로 설치된 것이다.

5. Import Assets

Capstone Repository의 Import 폴더 내의 파일들을 carla/Import 폴더에 복사 후 아래 명령어를 실행한다.

만약 Editor가 실행중이라면 종료 후 실행한다.

```
make import
```

Editor의 컨텐츠 브라우저의 루트 경로에, map_package가 생성되었다면 성공적으로 import된 것이다.

#### 기타 S/W 및 라이브러리 설치

- SUMO simulator <https://sumo.dlr.de/docs/Downloads.php>
- python 3.10.12 ?
- sumo 1.20.0
- numpy 1.26.4
  
이하는 pc환경 및 gpu 에 따라 버전을 맞추어 설치해야함
(아래는 Windows10, Nvidia GPU 8.6 (RTX 3060) 환경의 경우)

- cuda 11.8
- cuDNN 8.9.3
- pytorch 2.3.1

#### 모델 학습 및 테스트(추론) 방법

##### CARLA 시뮬레이터를 사용한 모델 학습 및 추론

**맵 지정**

컨텐츠 브라우저의 map_package의 cross_01.umap 파일을 연다.

맵을 연 상태에서, Alt+P 를 눌러, 시뮬레이션 환경을 실행한다.

명령 프롬프트에서 app 폴더로 이동한다.

```
cd app
```

**모델 학습**

```
python train.py --simulator carla     // 학습
```

실행 시 시뮬레이션 창에서 시뮬레이션이 시작되고, 카메라 뷰 창과 Total Reward 창이 켜진다.

1. 시뮬레이션 창
   랜덤하게 차량들이 생성되고 중앙의 교차로로 진입한다. 3600 tick마다 시뮬레이터가 초기화되어 또다른 랜덤한 환경이 재생된다.

2. 카메라 뷰 창
   북, 동, 남, 서 방향의 교차로 진입지점의 실시간 카메라 뷰가 나타난다. 각 뷰 하단에 현 교통 신호가 색깔로 나타나고 각 차선별 소요 프레임수의 총합(e.g. 349fr)이 출력된다.

   이미지상에 관심구역(ROI)가 선으로 표현되어 있으며 진입하는 각 차량을 둘러싼 바운딩박스가 그려진다.

   각 차량마다 id와 ROI 내부에 머물렀던 총 프레임수(소요 프레임 : lasted frames)가 레이블에 출력된다.

3. Total Reward 창
   Total Reward는 한 에피소드당 총 Reward(모든 차량의 소요프레임수의 합; 차량 대기시간과 비례)의 합에 음수를 취한 값이며, (이상적으로) 학습이 진행될수록 그래프가 서서히 증가 및 수렴한다.

Ctrl + C 시 학습 또는 추론이 중단되며, Total Reward 창을 닫으면 프로그램이 종료된다.

종료 시 학습이 완료된 모델, Total Reward 그래프와, 해당 Total Reward들의 실제 값을 기록한 텍스트파일이 아래 위치에 저장된다.

    app > results > carla > training
   

**모델 추론**

```
python control.py --simulator carla    // 추론
```

실행시 위의 모델 학습과 동일한 창들이 켜지며, 이번에는 미리 학습된 모델이 교차로의 신호를 제어하는 것을 볼 수 있다. 

이에 대한 Reward도 마찬가지로 Total Reward창에 나타난다.

Ctrl + C 시 학습 또는 추론이 중단되며, Total Reward 창을 닫으면 프로그램이 종료된다.

종료 시 Total Reward 그래프와 해당 Total Reward들의 실제 값을 기록한 텍스트파일이 아래 위치에 저장된다.

    app > results > carla > inference
    


##### SUMO 시뮬레이터를 사용한 모델 학습 및 추론


명령 프롬프트에서 app 폴더로 이동한다.

```
cd app
```

**모델 학습**

```
python train.py --simulator sumo     // 학습
```

실행 시 시뮬레이션 창과 Total Reward 창이 켜진다.

1. 시뮬레이션 창
   
   랜덤하게 차량들이 생성되고 중앙의 교차로로 진입한다. 1000 tick마다 시뮬레이터가 초기화되어 또다른 랜덤한 환경이 재생된다.

3. Total Reward 창
   Total Reward는 한 에피소드당 총 Reward(모든 차량의 대기시간의 합)의 합에 음수를 취한 값이며, (이상적으로) 학습이 진행될수록 그래프가 서서히 증가 및 수렴한다.


Ctrl + C 시 학습 또는 추론이 중단되며, Total Reward 창을 닫으면 프로그램이 종료된다.

종료 시 학습이 완료된 모델, Total Reward 그래프와, 해당 Total Reward들의 실제 값을 기록한 텍스트파일이 아래 위치에 저장된다.

    app > results > sumo > training
   

**모델 추론**

```
python control.py --simulator sumo      // 추론
```

실행 시 위의 모델 학습과 동일한 창들이 켜지며, 이번에는 미리 학습된 모델이 교차로의 신호를 제어하는 것을 볼 수 있다. 

이에 대한 Reward도 마찬가지로 Total Reward창에 나타난다.

Ctrl + C 시 학습 또는 추론이 중단되며, Total Reward 창을 닫으면 프로그램이 종료된다.

종료 시 Total Reward 그래프와 해당 Total Reward들의 실제 값을 기록한 텍스트파일이 아래 위치에 저장된다.

    app > results > sumo > inference



### 5. 팀 소개

**권오성**
- **DQN(Deep Q-Network) 알고리즘**을 효과적으로 교차로 제어 시나리오에 맞게 구현하기 위해 필수적인 구조적 설계 담당.
- CARLA 시뮬레이터와의 연동을 위한 **스켈레톤 구조**를 설계하고, 다양한 상태(state)와 행동(action)의 흐름이 자연스럽게 학습에 반영되도록 **코드 구조화**.
- 또한, 학습이 효율적으로 이루어지도록 DQN의 **하이퍼파라미터 튜닝**과 **메모리 관리 기법** 포함.

**이준표**
- CARLA의 다양한 **매개변수**를 설정하고 교차로 상황을 재현.
- 현실적인 시나리오를 구현했고, 이러한 시뮬레이션 결과가 **강화학습 알고리즘**에 실시간으로 전달되도록 데이터 흐름을 관리.
- 또한, 시뮬레이터의 **안정성 유지**를 위해 다양한 환경 설정 및 **성능 최적화**를 담당.

**정하립**
- **SUMO 시뮬레이터** 교차로 구축.
- SUMO 제어 스크립트 제작 및 DQN 코드와 통합.
- SUMO로 **FCNN과 CNN을 활용한 DQN 모델 학습** 진행 및 하이퍼파라미터 튜닝.
- CARLA 교차로 이미지로부터 **보상(reward)**를 도출하기 위해 YOLOv8과 ByteTracker를 활용한 **실시간 대기시간 계산 알고리즘** 제작.
- 전체 **코드 관리** 및 모델 **학습/테스트** 담당.
