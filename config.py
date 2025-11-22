# Transcript.py

MODEL_CONFIGS = {
    "large_v3": {
        "label": "Whisper Large v3 (Best Quality)",
        "model_name": "openai/whisper-large-v3",
        "info_text": "Highest accuracy, slowest processing speed."
    },
    "medium": {
        "label": "Whisper Medium (Balanced)",
        "model_name": "openai/whisper-medium",
        "info_text": "Good balance between speed and quality."
    },
    "small": {
        "label": "Whisper Small (Fast)",
        "model_name": "openai/whisper-small",
        "info_text": "Fastest processing speed, adequate quality."
    }
}

# Home.py
LOGO_FILENAME = "germany.png" # 로고 파일 이름
PAGE_TITLE = "Alyssa" # 페이지 제목
PAGE_ICON = "🎯" # 페이지 아이콘
PAGE_ROUTES = {
    "insight": {
        "label": " 1. Transcript",
        "path": "pages/1_Transcript.py"
    },
    "seeding": {
        "label": " 2. Seeding",
        "path": "pages/2_Seeding.py"
    },

}
SPACING_LARGE = 3
COLUMN_RATIO_LOGO = [1, 2, 1]
START = "START"
BACK = "← BACK"