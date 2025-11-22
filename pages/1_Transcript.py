import streamlit as st
from transformers import pipeline
import torch
import time
from config import MODEL_CONFIGS # Import config
import librosa
import numpy as np
import io

# -----------------------------------------------------------------
# 0. Page Configuration
# -----------------------------------------------------------------
st.set_page_config(
    page_title="AI Text Extractor",
    page_icon="📝",
    layout="wide"
)

st.title("Text Extractor 📝")
st.write("This app is dedicated to my hardworking gf") 

# -----------------------------------------------------------------
# 1. Model Selection UI
# -----------------------------------------------------------------
st.header("STEP 1: Choose Whisper Model") 

model_options_label = [config["label"] for config in MODEL_CONFIGS.values()]

selected_model_label = st.selectbox(
    "", 
    options=['--- select a model ---'] + model_options_label, 
    index=0, 
    placeholder="start by selecting a model",
    help="Large v3 model offers the best accuracy but takes longer to process. Medium model provides a good balance between speed and quality."
)

# Check if a real model is selected
is_model_selected = (selected_model_label != '--- select a model ---')

if is_model_selected:
    # Find the model key and info from the selected label
    selected_model_key = [k for k, v in MODEL_CONFIGS.items() if v["label"] == selected_model_label][0]
    selected_model_info = MODEL_CONFIGS[selected_model_key]

    st.info(f"✅ Model Selected: **{selected_model_info['label']}** - {selected_model_info['info_text']}")

    # -----------------------------------------------------------------
    # 2. Model Loading (STT Model Only!)
    # -----------------------------------------------------------------
    @st.cache_resource
    def load_stt_model(model_name):
        """ Load STT (Whisper) Model based on selection """
        start_time = time.time()
        st.info(f"Loading STT model: **{model_name}**...")
        
        # NOTE: device setting is handled by the canvas environment for execution
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # FIX: Removed the extra 'task' argument that caused the TypeError
        transcriber = pipeline(
            "automatic-speech-recognition", 
            model=model_name,
            device=device
        )
        
        loading_time = time.time() - start_time
        st.success(f"{model_name} model loaded successfully! - {loading_time:.2f}s")
        return transcriber

    transcriber = load_stt_model(selected_model_info["model_name"])
    st.divider()

    # -----------------------------------------------------------------
    # 3. Language & Chunking Selection UI
    # -----------------------------------------------------------------
    st.header("STEP 2: Configure Language & Chunking") 
    
    col1, col2 = st.columns([2, 1])

    with col1:
        language_option = st.radio(
            "1. Choose the primary language of your audio:",
            [
                "🌐 Multi-language (Auto-detect)",
                "🇰🇷 Korean only",
                "🇺🇸 English only",
                "🇯🇵 Japanese only",
                "🇨🇳 Chinese only",
                "🇷🇺 Russian only"
            ],
            index=0,
            key="lang_select",
            help="Select 'Multi-language' if your meeting contains mixed languages"
        )
        language_map = {
            "🌐 Multi-language (Auto-detect)": None, 
            "🇰🇷 Korean only": "korean",
            "🇺🇸 English only": "english",
            "🇯🇵 Japanese only": "japanese",
            "🇨🇳 Chinese only": "chinese",
            "🇷🇺 Russian only": "russian"
        }
        selected_language = language_map[language_option]
        if selected_language is None:
            st.info("✅ Auto-detect mode: All languages will be recognized automatically.")
        else:
            st.info(f"✅ Optimized for {selected_language.upper()} recognition.")
            
    with col2:
        # ** New Chunking Input **
        CHUNK_MINUTES = st.number_input(
            "2. Chunk Size (Minutes)",
            min_value=1,
            max_value=10, # Max 10 minutes chunk is reasonable
            value=3,
            step=1,
            help="The audio will be split into this many minutes for processing to improve speed and stability for large files."
        )


    st.divider()

    # -----------------------------------------------------------------
    # 4. File Upload UI
    # -----------------------------------------------------------------
    st.header("STEP 3: Upload Meeting Audio")
    uploaded_file = st.file_uploader(
        "Upload your audio file (mp3, m4a, wav, mp4, etc.)", 
        type=["mp3", "m4a", "wav", "mp4"]
    )

    # -----------------------------------------------------------------
    # 5. Main Logic (Chunked STT)
    # -----------------------------------------------------------------
    if uploaded_file is not None:
        
        st.audio(uploaded_file, format='audio/wav')
        audio_bytes = uploaded_file.read()
        full_text = ""
        
        # STT Logic
        st.info(f"Starting speech recognition with **{selected_model_info['label']}** and chunking by **{CHUNK_MINUTES} minutes**...")
        
        try:
            # Load audio using librosa from in-memory bytes
            # sr=16000 is standard for Whisper
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            duration_seconds = librosa.get_duration(y=audio, sr=sr)
            
            # Calculate chunk properties
            chunk_seconds = CHUNK_MINUTES * 60
            chunk_samples = chunk_seconds * sr
            
            # Total chunks
            num_chunks = int(np.ceil(len(audio) / chunk_samples))
            
            st.subheader(f"Total Duration: {duration_seconds:.2f} seconds | Chunks to Process: {num_chunks}")
            
            all_transcripts = []
            
            # Loop through chunks
            progress_bar = st.progress(0, text="Processing audio chunks...")
            
            for i in range(num_chunks):
                start_sample = i * chunk_samples
                end_sample = min((i + 1) * chunk_samples, len(audio))
                
                audio_chunk = audio[start_sample:end_sample]
                
                # Update progress bar
                progress_percent = (i + 1) / num_chunks
                progress_text = f"Processing chunk {i+1} of {num_chunks}..."
                progress_bar.progress(progress_percent, text=progress_text)
                
                # === 여기가 핵심 수정 부분 ===
                # Multi-language 모드일 때는 language 파라미터를 아예 안 넣어야 함!
                gen_kwargs = {"task": "transcribe"}
                
                # selected_language가 None이 아닐 때만 language 설정
                if selected_language is not None:
                    gen_kwargs["language"] = selected_language
                
                # Transcribe chunk
                chunk_result = transcriber(
                    audio_chunk, 
                    generate_kwargs=gen_kwargs,
                    return_timestamps=True
                )
                
                all_transcripts.append(chunk_result["text"])

            full_text = " ".join(all_transcripts)
            progress_bar.empty()
            st.success("🎉 Speech recognition complete!")

            # Translated subheader and text_area label
            st.subheader("📄 Full Meeting Transcript")
            st.text_area("STT Result", full_text, height=400)
            
        except Exception as e:
            st.error(f"Error during speech recognition or audio processing: {e}")
            st.exception(e)
            full_text = "" # Ensure no download button if error occurs

        # -----------------------------------------------------------------
        # 6. Download Button (If STT Succeeds)
        # -----------------------------------------------------------------
        if full_text:
            st.divider()
            st.header("STEP 4: Download Transcript") 
            
            lang_info = "Multi-language (Auto-detected)" if selected_language is None else selected_language.upper()
            model_name_for_report = selected_model_info['label']
            
            final_report = f"""
##############################################
# AI Meeting Transcript (Model: {model_name_for_report})
##############################################

[Source File: {uploaded_file.name}]
[Language Mode: {lang_info}]
[Chunk Size: {CHUNK_MINUTES} Minutes]

---

## [Full Meeting Transcript]
{full_text}
"""
            
            file_name_stem = uploaded_file.name.rsplit('.', 1)[0]
            
            st.download_button(
                label="✅ Download Full Transcript as .txt", 
                data=final_report,
                file_name=f"transcript_{file_name_stem}_{selected_model_key}_chunked.txt",
                mime="text/plain"
            )
            
else:
    st.warning("☝️ Please select a Whisper model")