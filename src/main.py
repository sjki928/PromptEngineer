import time
import random
from pathlib import Path

import streamlit as st
import torch
import numpy as np
from model import load_model_from_hf
from utils import get_prompt
from vllm import SamplingParams


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="LLM Prompt-Playground", page_icon="🤹")
st.header("Playground for prompt engineering", anchor="top", divider="rainbow")

# 시드 고정 함수
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

sampleparams=SamplingParams().update_from_generation_config({
  "max_tokens": 512,
  "stop_token_ids": 128001,
  "do_sample": True,
  "temperature": 0.3,
  "top_p": 0.9
    }
)

# UI 모델 설정
model_path = st.sidebar.text_input("Enter Hugging Face or Local Model Path (e.g., 'gpt2')", value="")
system_prompt = st.sidebar.text_area("System Prompt", value="This is the default system prompt.")

# 대화 로그 초기화 버튼 추가
if st.sidebar.button("Reset Chat Log"):
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# 대화 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# 모델 로딩
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None

# 새로운 모델 로딩 (새로운 모델 경로 입력 시)
if model_path and model_path != getattr(st.session_state, "last_model_path", None):
    # 새로운 모델을 로드할 때만
    st.session_state.llm_model, tokenizer = load_model_from_hf(model_path, existing_model=st.session_state.llm_model)
    st.session_state.last_model_path = model_path  # 입력된 모델 경로를 기록


# 이전 메시지 출력
for message in st.session_state.messages:
    if message["role"] != 'system':
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 사용자 입력 받기
if prompt := st.chat_input("Input yout message."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 모델 응답 처리
    if st.session_state.llm_model:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("처리 중..."):
                time.sleep(1)

            # system prompt와 사용자 메시지를 조합하여 프롬프트 생성
            prompt_text = get_prompt(st.session_state.messages)

            # 모델 응답 생성
            response = st.session_state.llm_model.generate(prompt_text,
                                                           sampling_params=sampleparams
                                                           )
            
            full_response = response[0].outputs[0].text.replace(prompt_text, "").strip()
            message_placeholder.markdown(full_response)
        
        # 모델 응답 저장
        st.session_state.messages.append({"role": "assistant", "content": full_response})
