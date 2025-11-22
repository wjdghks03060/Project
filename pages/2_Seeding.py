import streamlit as st
import pandas as pd
import numpy as np
import json
from openai import OpenAI
from fpdf import FPDF
import re
import time
import altair as alt

# ============================================================================
# 설정 및 상수
# ============================================================================

CONFIG = {
    'DATA_FILE': 'influencers_v25.csv', # (DB 연동으로 이제 사용되진 않음)
    'NUM_MOCK_ROWS': 10000,
    'DEFAULT_BENCHMARK_CPM': 15.0,
    'DEFAULT_BENCHMARK_CPE': 1.0,
    'API_MODEL': 'gpt-4-turbo-2024-04-09',
    'PDF_FONT_REGULAR': 'MALGUN.TTF',
    'PDF_FONT_BOLD': 'MALGUNBD.TTF',
    'MAX_ENGAGEMENT_RATE': 15.0 # [신규] 참여율 정규화를 위한 최대값
}

ALL_COUNTRIES = ['USA', 'Germany', 'Russia', 'France', 'UK', 'Japan', 'South Korea']
ALL_INTERESTS = sorted([
    'Skincare', 'K-Beauty', 'Makeup', 'Fashion', 'Lifestyle', 'Gaming', 'Tech',
    'Fitness', 'Wellness', 'Food', 'Travel', 'Music', 'K-Pop', 'Tiktok', 'Dance',
    'Vegan', 'Eco-friendly', 'Luxury', 'Minimalism', 'Art', 'Photography'
])
ALL_AGES = ['under_18', '18-24', '25-34', '35-44', '45-54', '55_plus'] 
ALL_GENDERS = ['Female', 'Male', 'Mixed']
ALL_PLATFORMS_CHOICES = ['Instagram', 'Tiktok', 'YouTube'] 

DEFAULT_SESSION_STATE = {
    # 상호작용 상태
    'clarification_phase': False,
    'clarification_data': None,
    'initial_campaign_goal': '',

    # 필터 상태 (가중치 기반)
    'target_countries': [],
    'country_weight': 1, 
    'target_ages': [],
    'age_weight': 1, 
    'target_genders': [],
    'gender_weight': 1, 
    'target_interests': [],
    'interest_weight': 1, 
    'target_platforms': [],
    'min_followers': 0,
    'max_followers': 1000000,
    
    # 신뢰도/효율성 가중치
    'engagement_weight': 1, 
    'fake_followers_weight': 1,

    # 기타
    'brand_keywords_input_4': [],
    'analysis_brand_guideline_input': '',
    'proposed_cost': '',
    'campaign_period': '',
    'content_guideline': '',
    'generated_contract': None,
    'brand_fit_result': None,
    'liked_influencers': set(),
    'insight_report': None,
    'filter_applied_success': False,
    'filter_error_message': None,
    'font_error': None
}

# ============================================================================
# 유틸리티 함수
# ============================================================================

def init_session_state():
    """세션 상태 초기화"""
    for key, default_value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# [신규] DB 연결 함수
@st.cache_resource
def get_db_connection():
    """Streamlit secrets를 사용해 MySQL DB 연결"""
    # .streamlit/secrets.toml의 [connections.mysql_db] 이름과 일치
    return st.connection("mysql_db", type="sql") 

@st.cache_resource
def get_openai_client():
    """OpenAI 클라이언트 싱글톤"""
    if "OPENAI_API_KEY" not in st.secrets:
        raise Exception("OpenAI API 키가 secrets.toml에 없습니다. (OPENAI_API_KEY)")
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_data
def to_csv(df):
    """데이터프레임을 CSV로 변환"""
    return df.to_csv(index=False).encode('utf-8-sig')

def to_float(value, default=0.0):
    """안전한 float 변환"""
    if value is None:
        return default
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return default
    try:
        clean_value = str(value).lower().replace('%', '').replace('k', '').strip()
        return float(clean_value)
    except (ValueError, TypeError):
        return default

def create_altair_bar_chart(data_series, title, sort_order=None, label_angle=0):
    """Pandas Series를 받아 Altair 차트를 생성하고 가독성을 높임"""
    df_chart = data_series.reset_index()
    df_chart.columns = ['Category', 'Value']

    if sort_order:
        x_sort = sort_order
    else:
        x_sort = alt.EncodingSortField(field="Value", op="sum", order='descending')
    
    y_title = 'Count'
    if 'Percentage' in title:
        y_title = 'Percentage (%)'
        
    chart = alt.Chart(df_chart).mark_bar().encode(
        x=alt.X('Category:N', sort=x_sort, title=None,
                axis=alt.Axis(labelAngle=label_angle, labelOverlap="greedy")),
        y=alt.Y('Value', title=y_title),
        tooltip=['Category', 'Value']
    ).properties(
        title=title,
        height=300
    )
    st.altair_chart(chart, use_container_width=True)

def format_follower_intervals(index):
    """팔로워 분포의 긴 숫자 구간을 'X K - Y K' 형태로 포맷팅"""
    formatted_index = []
    for interval_str in index:
        try:
            parts = str(interval_str).strip('[]()').split(', ')
            if len(parts) == 2:
                start = float(parts[0])
                end = float(parts[1])
                start_k = f"{start/1000:,.0f}K" if start >= 1000 else f"{start:,.0f}"
                end_k = f"{end/1000:,.0f}K" if end >= 1000 else f"{end:,.0f}"
                formatted_index.append(f"{start_k} - {end_k}")
            else:
                formatted_index.append(str(interval_str))
        except:
            formatted_index.append(str(interval_str))
    return formatted_index

def get_proposed_cost_suggestion(followers, estimated_cpm, estimated_cpe):
    """합리적인 제안 금액 산출"""
    bm_cpm = CONFIG['DEFAULT_BENCHMARK_CPM']
    estimated_exposure_cost = (followers / 1000) * estimated_cpm
    cpm_ratio = bm_cpm / estimated_cpm if estimated_cpm > 0 else 1.0 
    
    if cpm_ratio >= 1.2: # 매우 효율적
        proposal = estimated_exposure_cost * 1.5 
    elif cpm_ratio >= 0.8: # 적정 수준
        proposal = estimated_exposure_cost * 1.2 
    else: # 비효율적
        proposal = estimated_exposure_cost * 0.9 
        
    final_proposal = max(100, min(proposal, 5000))
    return f"${int(round(final_proposal, -1))} USD" 

# ============================================================================
# PDF 생성
# ============================================================================

class PDF(FPDF):
    def header(self):
        self.set_font('MalgunGothic', 'B', 15)
        self.cell(0, 10, 'AI 인플루언서 마케팅 보고서 (DRAFT)', 0, 1, 'C')
        self.ln(10)
    def chapter_title(self, title):
        self.set_font('MalgunGothic', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
    def chapter_body(self, body, font_size=10):
        self.set_font('MalgunGothic', '', font_size)
        try:
            self.multi_cell(0, 6, body)
        except Exception as e:
            self.multi_cell(0, 6, f"[텍스트 인코딩 오류: {str(e)[:50]}...]")
        self.ln()
    def add_korean_fonts(self):
        try:
            self.add_font('MalgunGothic', '', CONFIG['PDF_FONT_REGULAR'])
            self.add_font('MalgunGothic', 'B', CONFIG['PDF_FONT_BOLD'])
        except Exception:
            st.session_state.font_error = "PDF 한글 폰트(MALGUN.TTF) 로드 실패. Arial로 대체됩니다."
            self.set_font('Arial', '', 10)
            self.set_font('Arial', 'B', 10)

@st.cache_data
def generate_pdf_report(df_seeding_list, insight_report_content, brand_fit_result,
                        persona_context, filter_report_content, analysis_report_content,
                        influencer_name):
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.add_korean_fonts()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.chapter_title(f"인플루언서 마케팅 최종 분석 보고서 - {influencer_name}")
    pdf.chapter_title("1. AI 통합 분석 및 최종 제언")
    pdf.chapter_body(insight_report_content)
    pdf.ln(5)
    pdf.chapter_title("2. 브랜드 핏 평가 (Soft Fit)")
    if brand_fit_result:
        fit_score = brand_fit_result.get('brand_fit_score', 'N/A')
        fit_reason = brand_fit_result.get('reason', '분석 결과 없음')
        pdf.chapter_body(f"평가 점수: {fit_score} / 100점")
        pdf.chapter_body(f"평가 근거: {fit_reason}")
    else:
        pdf.chapter_body("브랜드 핏 분석이 수행되지 않았습니다.")
    pdf.ln(5)
    pdf.chapter_title("3. 캠페인 목표 및 분석 맥락")
    safe_persona = persona_context[:100] + "..." if len(persona_context) > 100 else persona_context
    safe_filter = filter_report_content[:100] + "..." if len(filter_report_content) > 100 else filter_report_content
    safe_analysis = analysis_report_content[:100] + "..." if len(analysis_report_content) > 100 else analysis_report_content
    pdf.chapter_body(f"캠페인 페르소나: {safe_persona}")
    pdf.chapter_body(f"1차 참고 보고서: {safe_filter}")
    pdf.chapter_body(f"추가 참고 보고서: {safe_analysis}")
    pdf.ln(5)
    pdf.chapter_title(f"4. 첨부 파일: 최종 시딩 후보군 목록 ({len(df_seeding_list)}명)")
    pdf.chapter_body("최종 시딩 후보군 목록 파일은 CSV로 별도 첨부됩니다.", font_size=8)
    return bytes(pdf.output(dest='S'))

# ============================================================================
# 데이터 생성 및 로딩 (*** MySQL 연동 수정 ***)
# ============================================================================

def create_mock_data(): # filename 인자 제거
    """[수정] MySQL DB에 가상 데이터 생성"""
    
    num_rows = CONFIG['NUM_MOCK_ROWS']
    
    with st.spinner(f"비율 기반 가상 데이터 생성 중... ({num_rows:,}건)"):
        def get_random_dist(options):
            dist = np.random.rand(len(options))
            dist /= dist.sum()
            return {option: round(val, 3) for option, val in zip(options, dist)}

        data = []
        for i in range(num_rows):
            row = {
                'influencer_name': f'influencer_{i}',
                'platform': np.random.choice(['Instagram', 'Tiktok', 'YouTube']),
                'followers': np.random.randint(10000, 1000000),
                'engagement_rate_pct': np.round(np.random.uniform(1.0, 10.0), 1),
                'fake_followers_pct': np.round(np.random.uniform(0.5, 30.0), 1)
            }

            age_dist = get_random_dist(ALL_AGES)
            for age_range, val in age_dist.items():
                row[f'age_{age_range}'] = val

            gender_dist = get_random_dist(ALL_GENDERS)
            for gender, val in gender_dist.items():
                row[f'gender_{gender}'] = val

            country_dist_dict = get_random_dist(ALL_COUNTRIES)
            interest_dist_dict = get_random_dist(ALL_INTERESTS)

            # DB 저장을 위해 JSON 문자열로 변환
            row['audience_country_dist'] = json.dumps(country_dist_dict)
            row['audience_interest_dist'] = json.dumps(interest_dist_dict)

            row['top_country'] = max(country_dist_dict, key=country_dist_dict.get)
            row['top_age_range'] = max(age_dist, key=age_dist.get)
            row['top_gender'] = max(gender_dist, key=gender_dist.get)
            row['top_interest'] = max(interest_dist_dict, key=interest_dist_dict.get)

            row['estimated_cpm'] = np.round(np.random.uniform(5.0, 50.0), 2)
            row['estimated_cpv'] = np.round(np.random.uniform(0.01, 0.50), 2)
            row['estimated_cpe'] = np.round(np.random.uniform(0.10, 2.00), 2)
            
            row['mock_brand_fit_score'] = np.random.randint(40, 100) 

            data.append(row)

        df = pd.DataFrame(data)
    
    # --- 이 부분이 핵심 수정 (DB에 저장) ---
    table_name = 'influencers_v25'
    with st.spinner(f"'{table_name}' 테이블에 데이터 {num_rows:,}건 INSERT 중... (기존 데이터 삭제)"):
        db_conn = get_db_connection()
        # SQLAlchemy 엔진을 사용하여 df.to_sql 실행
        # if_exists='replace' : 테이블이 이미 있으면 삭제하고 새로 만듦
        with db_conn.session as session:
            df.to_sql(name=table_name, con=session.bind, if_exists='replace', index=False)
            session.commit()
    
    st.success(f"'{table_name}' 테이블 생성 완료! ({num_rows:,}건)")

@st.cache_data
def load_data(): # filepath 인자 제거
    """[수정] MySQL DB에서 데이터 로드"""
    db_conn = get_db_connection()
    
    # 'influencers_v25'는 우리가 생성한 테이블 이름
    # ttl=600 : 10분(600초) 동안 쿼리 결과 캐시
    df = db_conn.query("SELECT * FROM influencers_v25;", ttl=600) 

    def safe_json_load(x):
        try:
            # DB에서 None으로 읽힐 경우 처리
            if x is None:
                return {}
            return json.loads(x)
        except (json.JSONDecodeError, TypeError):
            return {}

    # DB에서 JSON 문자열로 저장된 것을 다시 파싱
    df['country_dist_parsed'] = df['audience_country_dist'].apply(safe_json_load)
    df['interest_dist_parsed'] = df['audience_interest_dist'].apply(safe_json_load)
    
    if 'mock_brand_fit_score' not in df.columns:
         df['mock_brand_fit_score'] = 70 

    return df

# ============================================================================
# 필터링 로직 (Top-K 랭킹 모델) (*** 논리 수정 완료 ***)
# ============================================================================

def apply_platform_filter(df, target_platforms):
    """플랫폼 필터 적용 (멀티 선택 가능)"""
    if not target_platforms or df.empty:
        return df
    return df[df['platform'].isin(target_platforms)] 

def apply_numeric_filters(df, min_followers, max_followers):
    """숫자 범위 필터"""
    if df.empty:
        return df
    result = df[
        (df['followers'] >= min_followers) &
        (df['followers'] <= max_followers)
    ]
    return result

def get_filtered_influencers(df):
    """
    [논리 수정 완료] Top-K 랭킹 모델
    오디언스 적합도 항목은 타겟이 설정되었을 때만 가중치에 포함
    """
    result = df.copy()
    
    # 1. 하드 필터 적용 (플랫폼, 팔로워)
    result = apply_platform_filter(result, st.session_state.target_platforms) 
    result = apply_numeric_filters(result, 
                                   st.session_state.min_followers, 
                                   st.session_state.max_followers)
    
    if result.empty:
        return result

    # 2. '종합 적합도 점수' (Total_Fit_Score) 계산
    
    def calculate_dist_score(dist_dict, target_values):
        """타겟 값들의 오디언스 비율 합계를 계산 (0.0 ~ 1.0)"""
        if not isinstance(dist_dict, dict) or not target_values:
            return 0.0
        return sum(dist_dict.get(key, 0.0) for key in target_values)

    total_score = 0
    total_weight = 0
    
    # A. 오디언스 적합도 점수 (타겟이 설정된 경우에만 가중치 반영)
    
    # [수정됨] 1. 연령 (Age)
    if st.session_state.target_ages: 
        age_cols = [f'age_{age}' for age in st.session_state.target_ages]
        score_age = result[[col for col in age_cols if col in result.columns]].sum(axis=1) * 100
        total_score += (score_age * st.session_state.age_weight)
        total_weight += st.session_state.age_weight

    # [수정됨] 2. 성별 (Gender)
    if st.session_state.target_genders:
        gender_cols = [f'gender_{g}' for g in st.session_state.target_genders]
        score_gender = result[[col for col in gender_cols if col in result.columns]].sum(axis=1) * 100
        total_score += (score_gender * st.session_state.gender_weight)
        total_weight += st.session_state.gender_weight
        
    # [수정됨] 3. 국가 (Country)
    if st.session_state.target_countries: 
        score_country = result['country_dist_parsed'].apply(
            lambda d: calculate_dist_score(d, st.session_state.target_countries)
        ) * 100
        total_score += (score_country * st.session_state.country_weight)
        total_weight += st.session_state.country_weight
        
    # [수정됨] 4. 관심사 (Interest)
    if st.session_state.target_interests: 
        score_interest = result['interest_dist_parsed'].apply(
            lambda d: calculate_dist_score(d, st.session_state.target_interests)
        ) * 100
        total_score += (score_interest * st.session_state.interest_weight)
        total_weight += st.session_state.interest_weight
        
    # B. 신뢰도/효율성 점수 (항상 기본 가중치 1 이상으로 반영)
    if not result.empty:
        max_eng = CONFIG['MAX_ENGAGEMENT_RATE']
        
        # 5. 참여율 점수
        score_engagement = (result['engagement_rate_pct'].clip(0, max_eng) / max_eng) * 100
        total_score += (score_engagement * st.session_state.engagement_weight)
        total_weight += st.session_state.engagement_weight
        
        # 6. 가짜 팔로워 점수
        score_fake = (1 - (result['fake_followers_pct'].clip(0, 100) / 100)) * 100
        total_score += (score_fake * st.session_state.fake_followers_weight)
        total_weight += st.session_state.fake_followers_weight

    # C. 최종 점수 합산 및 정렬
    if total_weight > 0:
        result['Total_Fit_Score'] = (total_score / total_weight)
        result = result.sort_values(by=['Total_Fit_Score'], ascending=False)
    else:
        # (이론상 total_weight는 0이 될 수 없음. B 항목 때문에)
        result = result.sort_values(by='mock_brand_fit_score', ascending=False)
    
    return result

# ============================================================================
# AI 번역 함수 (수정 없음)
# ============================================================================

def translate_age_values(ai_ages):
    if not ai_ages: return []
    if isinstance(ai_ages, str): ai_ages = [ai_ages]
    result = []
    for age_str in ai_ages:
        age_lower = str(age_str).lower()
        if 'under_18' in age_lower or 'under 18' in age_lower: result.append('under_18')
        if '18-24' in age_lower or '18 to 24' in age_lower: result.append('18-24')
        if '25-34' in age_lower or '25 to 34' in age_lower: result.append('25-34')
        if '35-44' in age_lower or '35 to 44' in age_lower: result.append('35-44')
        if '45-54' in age_lower or '45 to 54' in age_lower: result.append('45-54')
        if '55_plus' in age_lower or '55+' in age_lower or '55 plus' in age_lower: result.append('55_plus')
        if '20대' in age_lower or '20s' in age_lower: result.extend(['18-24', '25-34'])
        if '30대' in age_lower or '30s' in age_lower: result.extend(['25-34', '35-44'])
        if '40대' in age_lower or '40s' in age_lower: result.extend(['35-44', '45-54'])
    return list(set(result))

def translate_platform_value(ai_platform):
    if not ai_platform: return []
    platforms = []
    platform_lower = str(ai_platform).lower()
    if "any" in platform_lower or "모두" in platform_lower or "all" in platform_lower:
        return ALL_PLATFORMS_CHOICES
    if "instagram" in platform_lower or "insta" in platform_lower: platforms.append('Instagram')
    if "tiktok" in platform_lower or "틱톡" in platform_lower: platforms.append('Tiktok')
    if "youtube" in platform_lower or "유튜브" in platform_lower: platforms.append('YouTube')
    if isinstance(ai_platform, list):
        for p in ai_platform:
            if 'instagram' in p.lower() and 'Instagram' not in platforms: platforms.append('Instagram')
            if 'tiktok' in p.lower() and 'Tiktok' not in platforms: platforms.append('Tiktok')
            if 'youtube' in p.lower() and 'YouTube' not in platforms: platforms.append('YouTube')
    return list(set(platforms)) 

# ============================================================================
# OpenAI API 호출 (*** 영어 버전으로 수정 ***)
# ============================================================================

def query_openai_for_clarification(initial_prompt, filter_report_content="N/A"):
    """[수정] AI가 사용자 입력 언어를 감지하고 해당 언어로 응답하도록 함"""
    client = get_openai_client()
    valid_countries = ", ".join(ALL_COUNTRIES)
    valid_interests = ", ".join(ALL_INTERESTS)
    valid_platforms = ", ".join(ALL_PLATFORMS_CHOICES)
    system_prompt = f"""
You are a marketing campaign assistant analyzing influencer seeding requirements.
Your task:
1. Understand the user's goal
2. Extract any filters you can identify immediately (e.g., target countries, ages, interests).
3. Generate 3-4 follow-up questions for missing critical information (e.g., Min Followers, Target Platform). **These questions must strictly ask for a specific value or descriptive information ONLY, and must NOT mention importance or weight.**
4. **IMPORTANT: You MUST detect the primary language of the user's input. All text in your JSON response (like 'understood' and 'follow_up_questions') MUST be in that same detected language.**

Valid Database Values:
- Countries: [{valid_countries}]
- Interests: [{valid_interests}]
- Ages: [under_18, 18-24, 25-34, 35-44, 45-54, 55_plus]
- Genders: [Female, Male, Mixed]
- Platforms: [{valid_platforms}]
Strategic Context (Reference Report Summary): {filter_report_content[:500]}
Respond in JSON format:
{{
  "understood": "Brief summary in the user's language",
  "follow_up_questions": [
    "Question 1 in the user's language?",
    "Question 2 in the user's language?"
  ],
  "initial_filters": {{
    "countries": [],
    "ages": [],
    "genders": [],
    "interests": [],
    "platform": "Any", 
    "min_followers_k": 0,
    "max_followers_k": 1000
  }}
}}
"""
    try:
        response = client.chat.completions.create(
            model=CONFIG['API_MODEL'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise Exception(f"OpenAI Clarification API Error: {e}")

def query_openai_refine_filters(initial_filters, follow_up_answers):
    # 이 함수는 언어와 크게 상관없으므로 원본 유지가능 (그래도 확인차 포함)
    client = get_openai_client()
    system_prompt = f"""
You are refining influencer search filters based on follow-up answers.
Update the initial filters with new information from answers.
**Do NOT set thresholds.** Only extract the filter values (e.g., countries, ages, min_followers_k).
Valid values same as before: Countries [{", ".join(ALL_COUNTRIES)}], etc.
Respond with updated JSON in same format as initial_filters.
"""
    user_prompt = f"""
Initial Filters: {json.dumps(initial_filters, ensure_ascii=False)}
Follow-up Answers:
{json.dumps(follow_up_answers, ensure_ascii=False)}
Please update and return the refined filters.
"""
    try:
        response = client.chat.completions.create(
            model=CONFIG['API_MODEL'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise Exception(f"OpenAI Refine API Error: {e}")

def query_openai_for_brand_fit(brand_keywords_str, audience_interest_json,
                                persona_input, brand_guideline_input, analysis_report_content):
    """[수정] AI가 사용자 입력(페르소나) 언어로 응답하도록 함"""
    client = get_openai_client()
    system_prompt = f"""
You are an expert brand marketer. Evaluate the semantic relevance (Soft Fit) between 
brand keywords and influencer audience interests.
**The 'reason' MUST be in the same language as the 'Brand Persona' input.**
Respond ONLY with JSON:
{{
  "brand_fit_score": <0-100>,
  "reason": "<one_sentence_in_the_user's_language>"
}}
"""
    user_prompt = f"""
- Strategic Context (Report Summary): "{analysis_report_content[:500]}"
- Brand Persona: "{persona_input}"
- Brand Guidelines: "{brand_guideline_input}"
- Brand Keywords: [{brand_keywords_str}]
- Audience Interest Distribution: {audience_interest_json}
Analyze and provide Brand Fit score.
"""
    try:
        response = client.chat.completions.create(
            model=CONFIG['API_MODEL'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise Exception(f"OpenAI Brand Fit API Error: {e}")

def query_openai_for_insight(influencer_data, brand_fit_result, context):
    """[수정] AI가 사용자 입력(캠페인 목표) 언어로 응답하도록 함"""
    client = get_openai_client()
    cpm = influencer_data['estimated_cpm']
    cpv = influencer_data['estimated_cpv']
    cpe = influencer_data['estimated_cpe']
    name = influencer_data['influencer_name']
    
    prompt = f"""
--- Context ---
**Campaign Goal:** "{context['persona']}"
**Market Report 1 (Filter Ref.):** "{context['filter_report'][:200]}..."
**Market Report 2 (Analysis Ref.):** "{context['analysis_report'][:200]}..."
--- Influencer Data ---
Name: {name}
CPM: ${cpm:.2f} (Benchmark: ${context['benchmark_cpm']:.2f})
CPV: ${cpv:.2f}
CPE: ${cpe:.2f} (Benchmark: ${context['benchmark_cpe']:.2f})
"""
    if brand_fit_result:
        prompt += f"""
Brand Fit: {brand_fit_result['brand_fit_score']}/100
Reason: {brand_fit_result['reason']}
"""
    prompt += f"""
--- Task ---
Write a comprehensive Strategic Insight Report **in the same language as the Campaign Goal** ("{context['persona']}").
The report must include:
1. Clear Recommendation
2. Cost Efficiency Analysis
3. Brand Fit Analysis (considering both reports)
4. Insights from reports
5. Final justification
"""
    try:
        response = client.chat.completions.create(
            model=CONFIG['API_MODEL'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI Insight API Error: {e}")

def query_openai_for_contract(influencer_name, proposed_cost,
                               campaign_period, content_guideline):
    """[수정] 계약서를 영어로 생성하도록 함"""
    client = get_openai_client()
    system_prompt = """
You are an AI legal assistant. Draft a professional influencer marketing contract 
**in English**. Start with the title: 'Influencer Marketing Campaign Agreement (Draft)'.
Use clear section headers.
"""
    user_prompt = f"""
Draft contract **in English**:
1. Influencer ("Contractor"): {influencer_name}
2. Brand ("Company"): d'Alba
3. Cost: {proposed_cost}
4. Period: {campaign_period}
5. Guidelines: {content_guideline}
"""
    try:
        response = client.chat.completions.create(
            model=CONFIG['API_MODEL'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI Contract API Error: {e}")

def query_openai_for_contract(influencer_name, proposed_cost,
                             campaign_period, content_guideline):
    client = get_openai_client()
    system_prompt = """
You are an AI legal assistant. Draft a professional influencer marketing contract 
in Korean. Start with title: '인플루언서 마케팅 캠페인 계약서 (초안)'.
Use clear section headers.
"""
    user_prompt = f"""
Draft contract in Korean:
1. Influencer (을): {influencer_name}
2. Brand (갑): d'Alba
3. Cost: {proposed_cost}
4. Period: {campaign_period}
5. Guidelines: {content_guideline}
"""
    try:
        response = client.chat.completions.create(
            model=CONFIG['API_MODEL'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI Contract API 오류: {e}")

# ============================================================================
# 콜백 함수 (수정 없음)
# ============================================================================

def apply_filters_from_ai(filters_data, is_skip=False):
    """AI가 추출한 필터를 세션 상태에 적용"""
    target_countries = filters_data.get('countries', [])
    target_ages = translate_age_values(filters_data.get('ages', []))
    target_genders = filters_data.get('genders', [])
    target_interests = filters_data.get('interests', [])
    target_platforms = translate_platform_value(filters_data.get('platform', []))
    
    st.session_state.target_countries = target_countries
    st.session_state.target_ages = target_ages
    st.session_state.target_genders = target_genders
    st.session_state.target_interests = target_interests
    st.session_state.target_platforms = target_platforms
    
    # AI가 타겟을 추출했다면, 해당 가중치를 3점으로 자동 설정
    st.session_state.country_weight = 3 if target_countries else 1
    st.session_state.age_weight = 3 if target_ages else 1
    st.session_state.gender_weight = 3 if target_genders else 1
    st.session_state.interest_weight = 3 if target_interests else 1
    
    st.session_state.min_followers = int(filters_data.get('min_followers_k', 0) * 1000)
    st.session_state.max_followers = int(filters_data.get('max_followers_k', 1000) * 1000)
    
    if filters_data.get('min_engagement_pct', 0) > 0:
        st.session_state.engagement_weight = 3
    if filters_data.get('max_fake_pct', 100) < 100:
        st.session_state.fake_followers_weight = 3


def unlike_influencer(name):
    """인플루언서 Like 해제"""
    st.session_state.liked_influencers.discard(name)
    if st.session_state.brand_fit_result and st.session_state.brand_fit_result[0] == name:
        st.session_state.brand_fit_result = None

def cb_skip_filters():
    """[콜백] 필터 건너뛰기"""
    data = st.session_state.clarification_data
    if data:
        apply_filters_from_ai(data['initial_filters'], is_skip=True)
        st.session_state.clarification_phase = False
        st.session_state.filter_applied_success = True

def cb_refine_filters():
    """[콜백] 필터 정제하기"""
    data = st.session_state.clarification_data
    if not data:
        return

    initial_filters = data['initial_filters']
    answers = {}
    for i, question in enumerate(data.get('follow_up_questions', [])):
        answer_key = f'followup_{i}'
        if answer_key in st.session_state and st.session_state[answer_key].strip():
            answers[f'question_{i}'] = st.session_state[answer_key]

    final_filters = initial_filters
    is_skip = False 

    if any(answers.values()):
        try:
            refined = query_openai_refine_filters(initial_filters, answers)
            final_filters = refined
        except Exception as e:
            st.session_state.filter_error_message = f"필터 정제 오류: {e}"
            final_filters = initial_filters 
            is_skip = True 

    apply_filters_from_ai(final_filters, is_skip=is_skip)
    st.session_state.clarification_phase = False
    st.session_state.filter_applied_success = True

# ============================================================================
# UI 렌더링 (*** 영어 버전으로 수정 ***)
# ============================================================================

def render_sidebar_expert_mode(df):
    """전문가 모드 사이드바 (가중치 중심)"""
    st.sidebar.header("🔧 Expert Mode (Weight Settings)")
    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Adjust the importance (Weight 1: Low ~ 5: Very High) for each category based on your campaign goals.")
    
    with st.sidebar.expander("🌍 Country", expanded=True):
        st.slider("Country Weight", 1, 5, st.session_state.country_weight, key='country_weight') 
        st.multiselect("Target Countries", ALL_COUNTRIES, key='target_countries')

    with st.sidebar.expander("🎂 Age", expanded=True):
        st.slider("Age Weight", 1, 5, st.session_state.age_weight, key='age_weight') 
        st.multiselect("Target Ages", ALL_AGES, key='target_ages',
                       format_func=lambda x: x.replace('_', ' ').title())

    with st.sidebar.expander("⚧️ Gender", expanded=False):
        st.slider("Gender Weight", 1, 5, st.session_state.gender_weight, key='gender_weight') 
        st.multiselect("Target Genders", ALL_GENDERS, key='target_genders')

    with st.sidebar.expander("🎨 Interest", expanded=False):
        st.slider("Interest Weight", 1, 5, st.session_state.interest_weight, key='interest_weight') 
        st.multiselect("Target Interests", ALL_INTERESTS, key='target_interests')

    st.sidebar.markdown("---")
    st.sidebar.caption("💡 Set the reliability and basic requirements for influencers.")

    with st.sidebar.expander("📊 Reliability", expanded=True):
        st.slider("Engagement Weight", 1, 5, st.session_state.engagement_weight, key='engagement_weight',
                   help="A higher score is given for higher engagement rates (0-15%).")
        st.slider("Fake Follower Weight", 1, 5, st.session_state.fake_followers_weight, key='fake_followers_weight',
                   help="A higher score is given for lower fake follower rates (0-100%).")
            
    with st.sidebar.expander("📱 Platform", expanded=False):
        st.multiselect("Target Platforms", ALL_PLATFORMS_CHOICES, key='target_platforms',
                       default=st.session_state.target_platforms) 

    with st.sidebar.expander("👥 Followers", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Min", min_value=0, step=10000, key='min_followers', format="%d")
        with col2:
            st.number_input("Max", min_value=0, step=10000, key='max_followers', format="%d", value=1000000)

def render_step1_conversational():
    """1단계: AI와 대화형 캠페인 기획"""
    st.subheader("1. 🤖 Conversational AI Campaign Planning")

    if st.session_state.get('filter_error_message'):
        st.error(st.session_state.filter_error_message)
        st.session_state.filter_error_message = None 

    # Phase 1: 초기 프롬프트 입력
    if not st.session_state.clarification_phase:
        st.markdown("💬 **What kind of influencers are you looking for?** Describe freely.")
        st.text_area(
            "Enter your campaign goal:",
            key='initial_campaign_goal',
            placeholder="e.g., Targeting 30s women in the US, promoting K-Beauty products...",
            height=150
        )
        st.file_uploader("📎 Reference Material (Optional, for filter context)", type=["txt", "pdf"], key='filter_report_file',
                         help="Attach market research reports, etc., for the AI to use as strategic context when setting filters.")

        if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            if not st.session_state.initial_campaign_goal.strip():
                st.error("Please enter your campaign goal.")
            else:
                with st.spinner("🤖 GPT-4 is analyzing your request..."):
                    try:
                        prompt = st.session_state.initial_campaign_goal
                        filter_report_content = "N/A"
                        uploaded_file = st.session_state.get('filter_report_file')
                        if uploaded_file:
                            if uploaded_file.type == "text/plain":
                                filter_report_content = uploaded_file.getvalue().decode("utf-8")[:1000]
                            else: 
                                filter_report_content = f"[{uploaded_file.name} file uploaded. Text parsing skipped.]"
                        
                        result = query_openai_for_clarification(prompt, filter_report_content)
                        st.session_state.clarification_data = result
                        st.session_state.clarification_phase = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI Analysis Error: {e}")

    # Phase 2: AI 추가 질문 & 답변
    else:
        data = st.session_state.clarification_data
        st.success(f"✅ **AI Understanding:** {data['understood']}")
        st.markdown("---")

        if data.get('follow_up_questions'):
            st.info("🤖 **Additional Questions for a More Accurate Search** (Optional)")
            for i, question in enumerate(data['follow_up_questions']):
                st.text_input(
                    f"**Q{i+1}.** {question}",
                    key=f'followup_{i}', 
                    placeholder="Enter answer or leave blank..."
                )
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.button("⚡ Skip & Search Now", use_container_width=True,
                           help="Search with the minimal info extracted by AI",
                           on_click=cb_skip_filters)
            with col2:
                st.button("✅ Submit Answers & Refined Search", type="primary",
                           use_container_width=True,
                           help="AI will refine filters based on your answers",
                           on_click=cb_refine_filters)
        else:
            st.success("✨ Your input is clear enough!")
            st.button("✅ Start Search", type="primary", use_container_width=True,
                      on_click=cb_skip_filters) 

        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.clarification_phase = False
            st.session_state.clarification_data = None
            st.session_state.initial_campaign_goal = ''
            if 'filter_error_message' in st.session_state:
                del st.session_state.filter_error_message
            if 'filter_applied_success' in st.session_state:
                del st.session_state.filter_applied_success
            st.rerun()

    # 성공 메시지
    if st.session_state.get('filter_applied_success'):
        st.success("✅ Filters applied! Adjust detailed weights in the left sidebar.")
        with st.expander("📊 View AI-configured filter details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Country:**", f"{st.session_state.target_countries or 'All'} (Weight {st.session_state.country_weight})")
                st.write("**Age:**", f"{st.session_state.target_ages or 'All'} (Weight {st.session_state.age_weight})")
                st.write("**Gender:**", f"{st.session_state.target_genders or 'All'} (Weight {st.session_state.gender_weight})")
                st.write("**Interest:**", f"{st.session_state.target_interests or 'All'} (Weight {st.session_state.interest_weight})")
            with col2:
                st.write("**Platform:**", f"{st.session_state.target_platforms or 'Any'}") 
                st.write("**Followers:**", f"{st.session_state.min_followers:,} ~ {st.session_state.max_followers:,}")
                st.write("**Engagement Weight:**", f"{st.session_state.engagement_weight}")
                st.write("**Fake Follower Weight:**", f"{st.session_state.fake_followers_weight}")
        st.session_state.filter_applied_success = False 

def render_step2_filtered_list(df):
    """2단계: 필터링된 인플루언서 목록"""
    st.subheader("2. 📋 Filtered Influencer List (Top 100)")

    filtered_df = get_filtered_influencers(df)
    
    active_filters = []
    if st.session_state.target_countries: active_filters.append(f"Country (W:{st.session_state.country_weight})")
    if st.session_state.target_ages: active_filters.append(f"Age (W:{st.session_state.age_weight})")
    if st.session_state.target_genders: active_filters.append(f"Gender (W:{st.session_state.gender_weight})")
    if st.session_state.target_interests: active_filters.append(f"Interest (W:{st.session_state.interest_weight})")
    active_filters.append(f"Engagement (W:{st.session_state.engagement_weight})") 
    active_filters.append(f"Reliability (W:{st.session_state.fake_followers_weight})") 
    sort_info = ["Campaign Fit Score (Weighted)"]
    
    st.info(f"**Fit Score Calculation Based On:** {' | '.join(active_filters) or 'Default Quality (Engagement, Reliability)'}")
    st.caption(f"**Sort By:** {' > '.join(sort_info)} (prioritized)")

    st.metric("Search Results", f"{len(filtered_df):,} / {len(df):,} total")

    if filtered_df.empty:
        st.warning("⚠️ No influencers match your filter criteria. Try broadening your conditions or adjusting them in the left sidebar.")
        return None

    st.markdown("**✅ Click the 'Select' checkbox to 'Like' influencers you are interested in:**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Like All (Top 100)", use_container_width=True):
            st.session_state.liked_influencers.update(filtered_df.head(100)['influencer_name'])
            st.rerun()
    with col2:
        if st.button("❌ Unlike All (Top 100)", use_container_width=True):
            st.session_state.liked_influencers.difference_update(filtered_df.head(100)['influencer_name'])
            st.rerun()

    filtered_df['✅ Select'] = filtered_df['influencer_name'].apply(
        lambda x: x in st.session_state.liked_influencers
    )

    cols_to_display = [
        '✅ Select', 'influencer_name', 'platform', 'followers',
        'engagement_rate_pct', 'fake_followers_pct', 
        'top_country', 'top_age_range', 'top_gender', 'top_interest'
    ]
    
    if 'Total_Fit_Score' in filtered_df.columns:
        filtered_df['Fit Score (100)'] = np.round(filtered_df['Total_Fit_Score'], 1)
        cols_to_display.insert(1, 'Fit Score (100)') 

    edited_df = st.data_editor(
        filtered_df[[col for col in cols_to_display if col in filtered_df.columns]].head(100),
        key='selection_editor',
        disabled=[col for col in cols_to_display if col != '✅ Select'],
        hide_index=True,
        height=400
    )

    # 동기화
    current_view_names = set(filtered_df.head(100)['influencer_name'])
    edited_likes = set(edited_df[edited_df['✅ Select'] == True]['influencer_name'])
    unliked = current_view_names - edited_likes
    
    st.session_state.liked_influencers.difference_update(unliked)
    st.session_state.liked_influencers.update(edited_likes)

    return filtered_df

def render_step3_detail_analysis(influencer_data, df):
    """3단계: 상세 분석 및 비용 확인"""
    st.subheader(f"3.1 {influencer_data['influencer_name']} (Basic Info)")

    st.markdown("#### 💰 Cost Metrics (Estimated)")
    bm_cpm = CONFIG['DEFAULT_BENCHMARK_CPM']
    bm_cpe = CONFIG['DEFAULT_BENCHMARK_CPE']
    cpm_color = "inverse" if influencer_data['estimated_cpm'] < bm_cpm else "off"
    cpe_color = "inverse" if influencer_data['estimated_cpe'] < bm_cpe else "off"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("CPM (Cost Per 1K Imp.)", 
                f"${influencer_data['estimated_cpm']:.2f}",
                delta=f"vs BM ${bm_cpm:.2f}",
                delta_color=cpm_color) 
    col2.metric("CPV (Cost Per View)", f"${influencer_data['estimated_cpv']:.2f}")
    col3.metric("CPE (Cost Per Engagement)", 
                f"${influencer_data['estimated_cpe']:.2f}",
                delta=f"vs BM ${bm_cpe:.2f}",
                delta_color=cpe_color)
    
    st.markdown("---")
    st.markdown("#### 📈 Audience Demographics")

    with st.expander("Click to view all distribution charts"):
        st.markdown("##### 🎂 Age Distribution")
        age_data = {col.split('_', 1)[1]: influencer_data[col] 
                    for col in df.columns if col.startswith('age_')}
        df_age = pd.DataFrame.from_dict(age_data, orient='index', columns=['Percentage'])
        df_age.index.name = 'Age Range'
        create_altair_bar_chart(df_age['Percentage'] * 100, 'Age Percentage', sort_order=ALL_AGES)

        st.markdown("##### ⚧️ Gender Distribution")
        gender_data = {col.split('_', 1)[1]: influencer_data[col] 
                       for col in df.columns if col.startswith('gender_')}
        df_gender = pd.DataFrame.from_dict(gender_data, orient='index', columns=['Percentage'])
        create_altair_bar_chart(df_gender['Percentage'] * 100, 'Gender Percentage')

        st.markdown("##### 🌍 Country Distribution (Top 5)")
        country_dist = influencer_data['country_dist_parsed']
        if country_dist:
            df_country = pd.DataFrame.from_dict(country_dist, orient='index', 
                                              columns=['Percentage']).nlargest(5, 'Percentage')
            create_altair_bar_chart(df_country['Percentage'] * 100, 'Country Percentage (Top 5)')
        else:
            st.write("No country data available")

        st.markdown("##### 🎨 Interest Distribution (Top 5)")
        interest_dist = influencer_data['interest_dist_parsed']
        if interest_dist:
            df_interest = pd.DataFrame.from_dict(interest_dist, orient='index', 
                                                columns=['Percentage']).nlargest(5, 'Percentage')
            create_altair_bar_chart(df_interest['Percentage'] * 100, 'Interest Percentage (Top 5)')
        else:
            st.write("No interest data available")

    st.markdown("---")
    st.button(f"💔 Unlike '{influencer_data['influencer_name']}'",
              type="secondary",
              use_container_width=True,
              on_click=lambda: unlike_influencer(influencer_data['influencer_name']),
              key="unlike_btn_3")

def render_step4_ai_analysis(influencer_data):
    """4단계: AI 심층 분석"""
    name = influencer_data['influencer_name']
    st.subheader(f"4.1 {name} (AI Deep Dive)")

    persona_context = st.session_state.initial_campaign_goal

    st.markdown("#### 4.2 Provide Analysis Context")
    if persona_context:
        st.info(f"**Applied Campaign Goal:**\n\n{persona_context}")
    else:
        st.warning("⚠️ Enter a campaign goal in Step 1 to improve AI analysis quality.")

    st.multiselect("A. Brand Core Keywords (for Soft Fit)",
                  options=ALL_INTERESTS,
                  key='brand_keywords_input_4',
                  help="Select keywords highly relevant to d'Alba")

    st.text_area("B. Brand Guidelines (Optional)",
                 key='analysis_brand_guideline_input',
                 placeholder="e.g., d'Alba emphasizes 'elegance', 'Italian', 'vegan'...")

    st.file_uploader("C. Additional Reference Report (for Analysis)", type=["txt", "pdf"], key='analysis_report_file',
                     help="Used as strategic context for generating brand fit and insight reports.")

    if st.button("Run GPT-4 Brand Fit Analysis", key="fit_btn_4"):
        brand_keywords = st.session_state.brand_keywords_input_4
        if not brand_keywords:
            st.error("Please select at least one brand keyword.")
        elif not persona_context:
            st.error("Please enter the campaign goal first.")
        else:
            with st.spinner("GPT-4 analyzing..."):
                try:
                    keywords_str = ", ".join(brand_keywords)
                    interest_json = json.dumps(influencer_data['interest_dist_parsed'])
                    
                    analysis_report_content = "N/A"
                    uploaded_file = st.session_state.get('analysis_report_file')
                    if uploaded_file:
                        if uploaded_file.type == "text/plain":
                            analysis_report_content = uploaded_file.getvalue().decode("utf-8")[:1000]
                        else:
                            analysis_report_content = f"[{uploaded_file.name} file uploaded. Text parsing skipped.]"

                    fit_result = query_openai_for_brand_fit(
                        keywords_str,
                        interest_json,
                        persona_context,
                        st.session_state.analysis_brand_guideline_input,
                        analysis_report_content 
                    )
                    st.session_state.brand_fit_result = (name, fit_result)
                    st.rerun()
                except Exception as e:
                    st.error(f"Brand Fit Analysis Error: {e}")

    if (st.session_state.brand_fit_result and
        st.session_state.brand_fit_result[0] == name):
        fit_data = st.session_state.brand_fit_result[1]
        st.markdown("#### 🎯 AI Brand Fit Score")
        st.metric("Score", f"{fit_data.get('brand_fit_score', 'N/A')} / 100")
        st.info(f"**AI Analysis:** {fit_data.get('reason', 'N/A')}")
    else:
        st.warning("Please click 'Run GPT-4 Brand Fit Analysis' first.")

    st.markdown("---")
    st.markdown("#### 4.3 Integrated Insight Report")
    
    if st.button("Generate GPT-4 Integrated Insight Report", type="primary", use_container_width=True):
        if not persona_context:
            st.error("You must enter a campaign goal first.")
        else:
            with st.spinner("GPT-4 performing integrated analysis..."):
                try:
                    filter_report = "N/A"
                    uploaded_filter_file = st.session_state.get('filter_report_file')
                    if uploaded_filter_file:
                        if uploaded_filter_file.type == "text/plain":
                            filter_report = uploaded_filter_file.getvalue().decode("utf-8")
                        else:
                            filter_report = f"[{uploaded_filter_file.name} file uploaded. Text parsing skipped.]"
                    
                    analysis_report = "N/A"
                    uploaded_analysis_file = st.session_state.get('analysis_report_file')
                    if uploaded_analysis_file:
                        if uploaded_analysis_file.type == "text/plain":
                            analysis_report = uploaded_analysis_file.getvalue().decode("utf-8")
                        else:
                            analysis_report = f"[{uploaded_analysis_file.name} file uploaded. Text parsing skipped.]"

                    brand_fit = None
                    if (st.session_state.brand_fit_result and
                        st.session_state.brand_fit_result[0] == name):
                        brand_fit = st.session_state.brand_fit_result[1]

                    context = {
                        'persona': persona_context,
                        'filter_report': filter_report,
                        'analysis_report': analysis_report,
                        'benchmark_cpm': CONFIG['DEFAULT_BENCHMARK_CPM'],
                        'benchmark_cpe': CONFIG['DEFAULT_BENCHMARK_CPE']
                    }

                    insight = query_openai_for_insight(influencer_data, brand_fit, context)
                    st.session_state.insight_report = (name, insight)
                    st.success(insight) # Show immediately after generation
                except Exception as e:
                    st.error(f"Insight Generation Error: {e}")

    # Show if report already exists
    if (st.session_state.insight_report and
        st.session_state.insight_report[0] == name):
        st.markdown("##### 💡 GPT-4 Final Analysis Report")
        st.success(st.session_state.insight_report[1])

    st.markdown("---")
    st.button(f"💔 Unlike '{name}'",
              type="secondary",
              use_container_width=True,
              on_click=lambda: unlike_influencer(name),
              key="unlike_btn_4")

def render_step5_contract_export(influencer_data, df):
    """5단계: 계약 및 결재"""
    name = influencer_data['influencer_name']
    st.subheader("5.1 Team Approval & Report Download")

    liked_list = list(st.session_state.liked_influencers)
    seeding_df = df[df['influencer_name'].isin(liked_list)].copy()

    st.info(f"Final candidates for approval: {len(seeding_df)} total")

    # CSV 다운로드
    st.markdown("#### A. Final Seeding List (CSV)")
    default_cols = ['influencer_name', 'platform', 'followers',
                    'estimated_cpm', 'estimated_cpe', 'top_country']
    available_cols = [col for col in default_cols if col in df.columns]
    cols_to_export = st.multiselect(
        "Select columns to export",
        options=list(df.columns),
        default=available_cols,
        key="export_cols"
    )
    if not seeding_df.empty and cols_to_export:
        export_df = seeding_df[cols_to_export]
        st.download_button(
            "✅ Download Final Seeding List (CSV)",
            data=to_csv(export_df),
            file_name="seeding_list.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")

    # PDF 다운로드
    st.markdown("#### B. AI Integrated Analysis Report (PDF)")
    if (st.session_state.insight_report and
        st.session_state.insight_report[0] == name):
        
        report_name, insight_content = st.session_state.insight_report
        
        filter_report = "N/A"
        uploaded_filter_file = st.session_state.get('filter_report_file')
        if uploaded_filter_file:
            if uploaded_filter_file.type == "text/plain": filter_report = uploaded_filter_file.getvalue().decode("utf-8")
            else: filter_report = f"[{uploaded_filter_file.name} file uploaded. Text parsing skipped.]"
        
        analysis_report = "N/A"
        uploaded_analysis_file = st.session_state.get('analysis_report_file')
        if uploaded_analysis_file:
            if uploaded_analysis_file.type == "text/plain": analysis_report = uploaded_analysis_file.getvalue().decode("utf-8")
            else: analysis_report = f"[{uploaded_analysis_file.name} file uploaded. Text parsing skipped.]"
        
        brand_fit = None
        if (st.session_state.brand_fit_result and st.session_state.brand_fit_result[0] == name):
            brand_fit = st.session_state.brand_fit_result[1]
            
        pdf_bytes = generate_pdf_report(
            seeding_df, insight_content, brand_fit,
            st.session_state.initial_campaign_goal,
            filter_report, analysis_report, name
        )
        st.download_button(
            f"✅ Download Integrated Report PDF (for {name})",
            data=pdf_bytes, file_name=f"report_{name}.pdf", mime="application/pdf", use_container_width=True
        )
    else:
        st.warning("You must first 'Generate Integrated Insight Report' in Step 4 to download the PDF.")

    st.divider()

    # 계약서 생성
    st.subheader("5.2 Draft Influencer Contract")
    st.info(f"Contract Target: **{name}**")
    
    suggested_cost = get_proposed_cost_suggestion(
        influencer_data['followers'], 
        influencer_data['estimated_cpm'], 
        influencer_data['estimated_cpe']
    )
    
    st.text_input("1. Proposed Cost (e.g., $500 USD)", key='proposed_cost', value=suggested_cost) 
    st.caption(f"💡 AI Suggestion: **{suggested_cost}** (Based on CPM/Followers)")
    
    st.text_input("2. Campaign Period (e.g., 2025-12-01 ~ 2025-12-15)", key='campaign_period')
    st.text_area("3. Content Guidelines", key='content_guideline',
                 placeholder="e.g., 1 TikTok video, 1 Instagram Reel by Dec 10th...")

    if st.button("Generate AI Contract Draft", type="primary", use_container_width=True):
        cost = st.session_state.proposed_cost
        period = st.session_state.campaign_period
        guideline = st.session_state.content_guideline
        
        if not all([cost, period, guideline]):
            st.error("All fields are required to generate the contract.")
        else:
            with st.spinner("GPT-4 drafting contract..."):
                try:
                    contract = query_openai_for_contract(name, cost, period, guideline)
                    st.session_state.generated_contract = (name, contract)
                    st.rerun()
                except Exception as e:
                    st.error(f"Contract Generation Error: {e}")

    if (st.session_state.generated_contract and
        st.session_state.generated_contract[0] == name):
        st.markdown("---")
        st.markdown("##### 💡 AI-Generated Contract Draft")

        contract_name, contract_text = st.session_state.generated_contract
        st.text_area("Generated Contract", value=contract_text, height=400, disabled=False)

        pdf_contract = PDF(orientation='P', unit='mm', format='A4')
        pdf_contract.add_korean_fonts() # (Note: This will use Arial if MALGUN fails)
        pdf_contract.add_page()
        pdf_contract.chapter_body(contract_text)
        pdf_bytes = bytes(pdf_contract.output(dest='S'))

        st.download_button(
            f"✍️ Download {contract_name} Contract Draft (PDF)",
            data=pdf_bytes,
            file_name=f"contract_{contract_name}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ============================================================================
# 메인 앱 (*** 영어 버전으로 수정 ***)
# ============================================================================

def run_app(df):
    """메인 애플리케이션 실행"""
    st.title("🚀 AI Influencer Seeding Platform")
    st.caption(f"V29 - MySQL Integrated (Data: {len(df):,} rows)")
    
    if st.session_state.get('font_error'):
        st.sidebar.warning(st.session_state.font_error)
        st.session_state.font_error = None 

    render_sidebar_expert_mode(df)

    render_step1_conversational()
    st.divider()

    filtered_df = render_step2_filtered_list(df)
    if filtered_df is None:
        return

    st.divider()

    liked_list = list(st.session_state.liked_influencers)

    if not liked_list:
        st.info("💡 First, 'Like' influencers in Step 2 using the '✅ Select' box.")
        return

    st.markdown("### Steps 3-5: Deep Dive on Selected Influencers")

    selected_name = st.selectbox(
        f"Select influencer to analyze (Total Liked: {len(liked_list)}):",
        options=liked_list,
        key="analysis_selector",
        help="View detailed info, AI analysis, and contracts for the selected influencer."
    )

    if not selected_name:
        st.info("Please select an influencer to analyze.")
        return

    influencer_data = df[df['influencer_name'] == selected_name].iloc[0]

    tab3, tab4, tab5 = st.tabs([
        "💰 3. Details & Cost",
        "💎 4. AI Deep Dive",
        "✍️ 5. Contract & Approval"
    ])

    with tab3:
        render_step3_detail_analysis(influencer_data, df)
    with tab4:
        render_step4_ai_analysis(influencer_data)
    with tab5:
        render_step5_contract_export(influencer_data, df)

# ============================================================================
# 메인 실행 (*** 영어 버전으로 수정 ***)
# ============================================================================

def main():
    st.set_page_config(
        layout="wide",
        page_title="d'Alba AI Seeding",
        page_icon="📦"   )

    init_session_state()

    tab_app, tab_data = st.tabs(["🚀 Seeding Platform", "⚙️ Data Management"])

    # PDF 폰트 로드 시도
    try:
        test_pdf = FPDF()
        test_pdf.add_font('MalgunGothic', '', CONFIG['PDF_FONT_REGULAR'])
    except Exception:
        # (이 경고는 그대로 둠 - 폰트 파일명 자체는 한국어)
        st.session_state.font_error = "⚠️ PDF Korean Font not set\n'MALGUN.TTF' file needed"

    with tab_data:
        st.subheader("📦 Data Management (MySQL)")
        
        df = None # df 변수 초기화
        try:
            # [수정] DB에서 데이터 로드
            df = load_data() 
            st.success(f"✅ 'influencers_v25' table loaded successfully ({len(df):,} rows)")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Influencers", f"{len(df):,}명")
                st.metric("Platforms", f"{df['platform'].nunique()} total")
            with col2:
                st.metric("Avg. Followers", f"{df['followers'].mean():,.0f}")
                st.metric("Avg. Engagement Rate", f"{df['engagement_rate_pct'].mean():.2f}%")

            if st.button("🔄 Regenerate Mock Data (Overwrite DB)", use_container_width=True):
                st.cache_data.clear() # 캐시 비우기
                st.cache_resource.clear() # DB 커넥션 캐시도 비우기
                create_mock_data() # [수정] DB에 새로 생성
                st.rerun()

            with st.expander("📊 Data Preview (DB)"):
                st.dataframe(df.head(20), use_container_width=True)

            with st.expander("📈 Data Statistics (DB)"):
                st.write("**Platform Distribution:**")
                create_altair_bar_chart(df['platform'].value_counts(), 'Platform Distribution')

                st.write("**Follower Distribution:**")
                follower_dist = pd.cut(df['followers'], bins=5).value_counts().sort_index()
                follower_dist.index = format_follower_intervals(follower_dist.index)
                create_altair_bar_chart(follower_dist, 'Follower Distribution')

        except Exception as e:
            # [수정] FileNotFoundError가 아닌 DB 연결/테이블 없음 오류 처리
            st.warning(f"⚠️ Could not find 'influencers_v25' table in DB or an error occurred.")
            st.error(f"Error Details: {str(e)[:200]}...") # 실제 오류 메시지 표시
            st.info("Press the button below to generate mock data in the database.")

            if st.button("📦 Generate Mock Data in DB (10,000 rows)",
                           use_container_width=True,
                           type="primary"):
                st.cache_data.clear()
                st.cache_resource.clear()
                create_mock_data() # [수정] DB에 생성
                st.rerun()
            
            df = None # 오류 발생 시 df는 None으로 유지

    with tab_app:
        if df is not None:
            # DB 스키마가 올바른지 간단히 확인
            required_cols = ['age_under_18', 'top_country', 'country_dist_parsed']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                st.error(f"⚠️ Data Format Error. Missing columns: {missing_cols}")
                st.info("Please regenerate data in the '⚙️ Data Management' tab.")
            else:
                run_app(df)
        else:
            st.info("👈 Please generate or load data in the '⚙️ Data Management' tab first.")
            st.markdown("""
            ### 💡 How to Use
            
            1.  **Data Management Tab**: Generate mock data (in DB).
            2.  **Conversational AI**: Enter your campaign goal.
            3.  GPT-4 will generate follow-up questions.
            4.  Answer or skip to start the search.
            5.  **Expert Mode (Left)**: Fine-tune the search.
            6.  **'Like'** influencers you're interested in.
            7.  Run AI analysis and generate contracts.
            
            ### ✨ Key Features
            
            - 🤖 **Conversational AI**: Natural language input.
            - ⚡ **MySQL Integration**: Real-time database connection.
            - 🔧 **Expert Mode**: Weight-based Top-K ranking.
            - 💎 **Brand Fit Analysis**: GPT-4 based scoring.
            - 📄 **Auto-Contract**: AI drafts contracts automatically.
            """)

if __name__ == "__main__":
    main()