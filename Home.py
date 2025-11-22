import streamlit as st
from PIL import Image
# config.py 파일에서 필요한 설정 값을 모두 불러옵니다.
from config import PAGE_TITLE, PAGE_ICON, PAGE_ROUTES, LOGO_FILENAME, SPACING_LARGE, COLUMN_RATIO_LOGO ,START, BACK
# 페이지 설정
st.set_page_config(
    page_title= PAGE_TITLE,
    page_icon= PAGE_ICON,
    layout="centered"
)
# 세션 스테이트 초기화 (페이지 관리를 위해)
if 'page' not in st.session_state:
    st.session_state.page = 'home'
# ✨ 새로 추가: 중앙 레이아웃 함수
def centered_layout(ratio=COLUMN_RATIO_LOGO):
    """중앙 정렬 레이아웃 생성 후 가운데 컬럼 반환"""
    col1, col2, col3 = st.columns(ratio)
    return col2
# --- 1. 홈 페이지 함수 ---
def home_page():
    # 여백 추가
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    # 로고를 중앙에 배치
    col = centered_layout() 

    with col:      
        try:
            logo = Image.open(LOGO_FILENAME) 
            st.image(logo, use_container_width=True)
        except FileNotFoundError:
            # 로고 파일이 없을 경우 텍스트로 표시
            st.markdown(
                "<h1 style='text-align: center; font-size: 72px; color: #4CAF50;'>Alyssa</h1>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"로고 로딩 오류: {e}")
            st.markdown(
                "<h1 style='text-align: center; font-size: 72px; color: #4CAF50;'>Alyssa</h1>",
                unsafe_allow_html=True
            )
    # 여백 추가
    st.markdown("<br>" * SPACING_LARGE, unsafe_allow_html=True)
    # 버튼을 중앙에 배치
    col = centered_layout()
    with col:
        if st.button(START, type="primary", use_container_width=True):
            st.session_state.page = 'select' 
            st.rerun()
# --- 2. 페이지 선택 함수  ---
def select_page():
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 딕셔너리를 반복문으로 돌면서 버튼 생성 (Configuration 사용)
    for key, info in PAGE_ROUTES.items(): # <-- 이 반복문이 5개의 if 버튼을 대체해
        if st.button(info["label"], use_container_width=True):
            try:
                st.switch_page(info["path"]) 
            except Exception as e:
                st.error(f"페이지 이동 오류: {e}")
                st.info(f"'{info['path']}' 파일 경로와 이름을 확인하세요.")
    st.markdown("---")   
    # '홈으로 돌아가기' 버튼
    if st.button(BACK):
        st.session_state.page = 'home'
        st.rerun()
# --- 3. 페이지 라우팅 ---
# 세션 상태에 따라 표시할 함수를 결정
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'select':
    select_page()
else:
    home_page() # 기본값