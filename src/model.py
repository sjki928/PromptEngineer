# model_utils.py
import torch, gc
from transformers import AutoTokenizer
from vllm import LLM
import contextlib
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import streamlit as st

# 기존 모델을 해제하는 함수
def unload_model(model):
    if model:
        destroy_model_parallel()
        destroy_distributed_environment()
        del model.llm_engine.model_executor
        del model
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()  # 캐시 메모리를 해제하여 메모리 확보

# 모델 로딩 함수
def load_model_from_hf(model_name, existing_model=None):
    try:
        # 기존 모델 해제
        unload_model(existing_model)

        st.sidebar.write("Loading model from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LLM(model=model_name)  # vLLM을 통해 Hugging Face 모델 로드
        return model, tokenizer
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None, None
