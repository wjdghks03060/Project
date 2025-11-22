# 📝 AI Text Extractor (Whisper STT Tool)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Whisper](https://img.shields.io/badge/Model-Whisper-lightgrey.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

**AI Text Extractor**는 OpenAI의 Whisper 모델을 활용하여 긴 회의, 강의, 인터뷰 녹음 파일을 텍스트로 변환하는 고성능 STT(Speech-to-Text) 솔루션입니다.
단순한 변환을 넘어, 긴 오디오 파일을 안정적으로 처리하기 위한 **스마트 청크(Chunk) 프로세싱** 기술이 적용되어 GPU 메모리 부족(OOM) 없이 안정적인 변환을 지원합니다.

> *"This app is dedicated to my hardworking gf"*

## 📋 목차
1. [주요 기능 (Key Features)](#-주요-기능-key-features)
2. [시스템 아키텍처](#-시스템-아키텍처-system-architecture)
3. [기술 스택 (Tech Stack)](#-기술-스택-tech-stack)
4. [설치 및 실행 (Installation)](#-설치-및-실행-installation)
5. [사용 방법 (Usage Guide)](#-사용-방법-usage-guide)

---

## ✨ 주요 기능 (Key Features)

### 1. 🎯 맞춤형 Whisper 모델 선택
- **Model Flexibility:** 사용자의 하드웨어 사양과 필요 정확도에 따라 모델을 선택할 수 있습니다.
    - `openai/whisper-large-v3`: 최고 수준의 정확도 (다국어 인식 권장)
    - `openai/whisper-medium/small`: 속도와 성능의 균형
    - `distil-whisper`: 빠른 처리 속도에 최적화

### 2. ✂️ 스마트 청크 프로세싱 (Smart Chunking)
- **Memory Safety:** 1시간 이상의 긴 오디오 파일도 한 번에 로드하지 않고, `Librosa`를 통해 설정된 시간(예: 3분) 단위로 분할하여 처리합니다.
- **Stability:** 분할 처리 로직을 통해 저사양 GPU 환경에서도 긴 회의록 작성이 가능합니다.

### 3. 🌍 강력한 다국어 지원
- **Auto-Detect:** 언어가 혼재된 회의의 경우 자동 감지 모드(`Multi-language`)를 지원합니다.
- **Language Force:** 한국어, 영어, 일본어, 중국어, 러시아어 등 특정 언어로 인식을 고정하여 정확도를 극대화할 수 있습니다.

### 4. 📄 메타데이터 포함 자동 리포트
- 변환된 스크립트뿐만 아니라 **사용된 모델, 언어 설정, 청크 사이즈, 원본 파일명** 등의 메타데이터가 포함된 `.txt` 포맷의 리포트를 자동으로 생성 및 다운로드합니다.

---

## 🏗 시스템 아키텍처 (System Architecture)

1. **Audio Input:** `mp3`, `wav`, `m4a`, `mp4` 등 다양한 포맷 지원
2. **Preprocessing:** `Librosa`를 이용해 16,000Hz로 리샘플링 및 Numpy Array 변환
3. **Chunking Logic:** 설정된 분(Minute) 단위로 오디오 데이터 슬라이싱
4. **AI Inference:** HuggingFace `transformers` 파이프라인을 통해 청크별 추론 (GPU 가속)
5. **Output Gen:** 텍스트 결합 및 최종 리포트 생성

---

## 🛠 기술 스택 (Tech Stack)

- **Frontend:** Streamlit (UI/UX)
- **AI Core:** OpenAI Whisper, HuggingFace Transformers
- **Audio Processing:** Librosa, FFMPEG
- **Compute:** PyTorch (CUDA Support)
- **Data Handling:** NumPy, IO

---

## ⚡ 설치 및 실행 (Installation)

### 1. 필수 시스템 요구사항 (FFMPEG)
오디오 처리를 위해 반드시 FFMPEG가 설치되어 있어야 합니다.
- **Windows:** [설치 가이드](https://wikidocs.net/15223) (환경 변수 Path 설정 필수)
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

### 2. Python 패키지 설치
```bash
# 가상환경 활성화 후
pip install streamlit transformers librosa numpy torch
# (GPU 사용 시 PyTorch는 CUDA 버전으로 설치 권장)
```
### 3. 애플리케이션 실행
```bash
streamlit run pages/1_Transcript.py
```

## 📖 사용 방법 (Usage Guide)

1. **Step 1 (Model):** 드롭다운에서 원하는 Whisper 모델을 선택합니다. (최초 실행 시 모델 다운로드 소요)
2. **Step 2 (Config):**
   - **주 언어(Primary Language)**를 선택합니다. (한국어 회의는 `Korean only` 권장)
   - **청크 사이즈(Chunk Size)**를 설정합니다. (기본 3분 권장)
3. **Step 3 (Upload):** 회의 녹음 파일을 업로드합니다.
4. **Step 4 (Download):** 변환 진행률을 확인하고, 완료 후 생성된 텍스트 파일을 다운로드합니다.