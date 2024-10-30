# PNU_Graduation_2024
PNU_Graduation_2024

### 1. 프로젝트 소개

프로젝트 명, 목적, 개요 등 프로젝트에 대한 간단한 소개글을 작성하세요.

### 2. 팀 소개

프로젝트에 참여한 팀원들의 이름, 이메일, 역할를 포함해 팀원들을 소개하세요.

### 3. 구성도

프로젝트 결과물의 개괄적인 동작을 파악할 수 있는 이미지와 글을 작성하세요.

### 4. 소개 및 시연 영상

프로젝트 소개나 시연 영상을 넣으세요.

### 5. 사용법

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

#### 시뮬레이터 사용법

**맵 지정**

컨텐츠 브라우저의 map_package의 cross_01.umap 파일을 연다.

맵을 연 상태에서, Alt+P 를 눌러, 시뮬레이션 환경을 실행한다.

Anaconda와 같은 가상 환경에서 아래 명령을 실행할 수 있다.

```
cd app
python train.py     // 학습
python control.py   // 추론
```