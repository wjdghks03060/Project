# 🚀 AI Influencer Seeding Platform (d'Alba Custom)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-green.svg)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-red.svg)

**AI 인플루언서 시딩 플랫폼**은 브랜드(d'Alba)의 마케팅 캠페인 목표에 최적화된 인플루언서를 발굴, 분석, 계약하는 전 과정을 자동화한 올인원 솔루션입니다.
자연어 기반의 캠페인 기획, **MySQL 실시간 데이터 연동**, 그리고 GPT-4를 활용한 **심층 분석 및 계약서 자동 생성** 기능을 통합하여 마케팅 효율을 극대화합니다.

## 📋 목차
1. [주요 기능 (Key Features)](#-주요-기능-key-features)  
2. [시스템 아키텍처](#-시스템-아키텍처)             
3. [기술 스택 (Tech Stack)](#-기술-스택-tech-stack)   
4. [설치 및 설정 (Setup)](#-설치-및-설정-setup)        
5. [사용 방법 (Usage Guide)](#-사용-방법-usage-guide)

---

## <a id="key-features"></a>✨ 주요 기능 (Key Features)

### 1. 🤖 대화형 AI 캠페인 기획 (Conversational Planning)
- **Natural Language Understanding:** "30대 미국 여성을 타겟으로 하는 비건 화장품 캠페인"과 같이 문장으로 입력하면, AI가 의도를 분석합니다.
- **Smart Query Generation:** AI가 국가, 연령, 관심사 등의 필터 조건을 자동으로 추출하고, 부족한 정보에 대해 역질문(Follow-up Question)을 생성합니다.

### 2. 📦 MySQL 데이터베이스 연동 & 관리
- **Real-time DB Connection:** 로컬 CSV 파일 방식이 아닌, MySQL DB와 직접 연동되어 대용량 데이터를 실시간으로 조회합니다.
- **Mock Data Generator:** 시스템 테스트를 위해 `create_mock_data()` 기능을 내장, 10,000건 이상의 정교한 가상 인플루언서 데이터를 생성하여 DB에 적재합니다.

### 3. 🔧 전문가 모드 (Expert Mode) & Top-K 랭킹
- **Weighted Scoring System:** 국가, 연령, 성별, 관심사, 참여율, 신뢰도(가짜 팔로워) 등 각 항목에 가중치(1~5점)를 부여하여 **Total Fit Score**를 산출합니다.
- **Dynamic Filtering:** 오디언스 분포 비율까지 고려한 정교한 알고리즘으로 최적의 후보군을 랭킹 순으로 추천합니다.

### 4. 💎 AI 브랜드 핏 분석 (Deep Dive Analysis)
- **Soft Fit Evaluation:** 브랜드 키워드와 인플루언서의 오디언스 성향을 GPT-4가 의미론적으로 분석하여 100점 만점의 적합도 점수를 산출합니다.
- **Insight Report:** 업로드된 시장 조사 리포트(PDF/TXT)를 문맥으로 활용하여, 해당 인플루언서를 선정해야 하는 전략적 이유가 담긴 리포트를 자동 생성합니다.

### 5. 📄 자동 계약 및 리포트 (Auto-Contract)
- **Smart Contract Drafting:** 협의된 비용, 기간, 가이드라인을 입력하면 법적 효력이 있는 수준의 계약서 초안(한글/영문)을 AI가 작성합니다.
- **PDF Export:** AI 분석 리포트와 생성된 계약서를 PDF 파일로 즉시 다운로드할 수 있습니다.

---

## <a id="architecture"></a>🏗 시스템 아키텍처

1. **User Interface:** Streamlit 기반의 대화형 웹 인터페이스
2. **Logic Layer:** Python 기반의 필터링 로직 (Top-K Ranking)
3. **Data Layer:** MySQL (AWS RDS or Local) + SQLAlchemy ORM
4. **Intelligence Layer:** OpenAI GPT-4 Turbo (Analysis, Chat, Drafting)

---

## <a id="tech-stack"></a>🛠 기술 스택 (Tech Stack)

- **Language:** Python 3.10+
- **Framework:** Streamlit
- **Database:** MySQL, SQLAlchemy (Connection Pooling)
- **AI API:** OpenAI API (GPT-4 Turbo)
- **Visualization:** Altair (Interactive Charts)
- **Reporting:** FPDF (PDF Generation)
- **Libraries:** Pandas, NumPy, Pymysql

---

## <a id="setup"></a>⚡ 설치 및 설정 (Setup)

### 1. 데이터베이스 준비
MySQL 서버가 실행 중이어야 하며, `secrets.toml`에 연결 정보를 입력해야 합니다.

### 2. Secrets 설정 (`.streamlit/secrets.toml`)
프로젝트 루트의 `.streamlit` 폴더에 `secrets.toml` 파일을 생성하고 아래 내용을 반드시 설정해야 합니다.

```toml
# OpenAI API Key
OPENAI_API_KEY = "sk-..."

# MySQL Connection (SQLAlchemy URL 형식)
[connections.mysql_db]
url = "mysql+pymysql://[DB유저명]:[비밀번호]@[호스트IP]:3306/[DB이름]"
# 예시: url = "mysql+pymysql://admin:password123@localhost:3306/influencer_db"
```

### 3. 폰트 파일 설정
PDF 리포트의 한글 깨짐 방지를 위해 프로젝트 루트 폴더에 다음 폰트 파일이 존재해야 합니다.
- `MALGUN.TTF` (일반)
- `MALGUNBD.TTF` (볼드)

### 4. 패키지 설치 및 실행
```bash
pip install -r requirements.txt
streamlit run Home.py
```
## <a id="usage"></a>📖 사용 방법 (Usage Guide)

1. **Step 0 (Data):** 'Data Management' 탭에서 'Regenerate Mock Data'를 클릭하여 DB를 초기화합니다.
2. **Step 1 (AI Plan):** 캠페인 목표를 입력하고 AI와 대화하며 타겟 조건을 설정합니다.
3. **Step 2 (Select):** 추출된 리스트에서 'Expert Mode' 가중치를 조절하고, 후보군을 선택(Like)합니다.
4. **Step 3 (Analyze):** 선택한 인플루언서 탭에서 'Brand Fit Analysis'를 실행하여 심층 분석을 확인합니다.
5. **Step 4 (Contract):** 제안 비용과 기간을 입력하여 계약서를 생성하고 PDF로 다운로드합니다.