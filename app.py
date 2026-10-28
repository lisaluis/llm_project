import os
import gc
import time
import threading
from contextlib import nullcontext

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTERS = {
    "SFT (QLoRA)": ("sft", os.getenv("SFT_ADAPTER_PATH", "./models/qwen_qlora_sft_100")),
    "SFT → ORPO": ("orpo", os.getenv("ORPO_ADAPTER_PATH", "./models/qwen_orpo_adapter")),
}
CONFIGS = ["Base", "SFT (QLoRA)", "SFT → ORPO"]
HAS_CUDA = torch.cuda.is_available()
CPU_DTYPE = getattr(torch, os.getenv("CPU_DTYPE", "bfloat16"), torch.bfloat16)
SYSTEM_DEFAULT = ("You are a supportive assistant. Be empathetic, practical, non-judgmental, "
                  "and do not diagnose or replace professional care.")

st.set_page_config(page_title="Mental Health LLM Lab", page_icon="🫶", layout="wide")
st.title("Mental Health LLM Lab")
st.caption("Research prototype | Qwen2.5-1.5B-Instruct | QLoRA SFT → ORPO")

with st.sidebar:
    st.header("Experiment parameters")
    model_choice = st.selectbox("Model configuration", CONFIGS)
    if HAS_CUDA:
        quantization = st.selectbox("Inference quantization", ["4-bit NF4", "8-bit", "None"])
    else:
        quantization = "None"
        st.info("CPU server detected: quantization is disabled and the model runs in "
                f"{CPU_DTYPE}. Expect slow generation (seconds per token).")
    max_new_tokens = st.slider("Maximum new tokens", 32, 512, 128, 16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 100, 50, 1)
    do_sample = st.checkbox("Sampling enabled", value=True)
    st.divider()
    st.subheader("Training configuration (reference)")
    st.caption("These settings describe the saved experiment. Changing them here does not retrain the model.")
    st.code("SFT: NF4, r=8, alpha=16, dropout=0.05, epochs=2, LR=2e-4, max_seq_len=512\n"
            "ORPO: 20 preference pairs, 1 epoch, LR=5e-6, beta=0.1")
    st.warning("Not a diagnostic or treatment tool. If someone may be in immediate danger, "
               "contact local emergency services or a trusted person who can help now.")


@st.cache_resource(show_spinner="Loading base model. The first load downloads ~3 GB and can take several minutes...")
def load_engine(quant_choice):
    """Load the base model ONCE. Adapters are attached to this single copy (saves RAM)."""
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    kwargs = {"low_cpu_mem_usage": True}
    if HAS_CUDA:
        from transformers import BitsAndBytesConfig
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
        if quant_choice == "4-bit NF4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        elif quant_choice == "8-bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = CPU_DTYPE
        torch.set_num_threads(os.cpu_count() or 1)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    model.eval()
    return {"tok": tok, "model": model, "peft": None, "lock": threading.Lock()}


def select_model(engine, choice):
    """Return (model, context manager) for the requested configuration."""
    peft_model = engine["peft"]
    if choice == "Base":
        if peft_model is None:
            return engine["model"], nullcontext()
        return peft_model, peft_model.disable_adapter()
    name, path = ADAPTERS[choice]
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        raise FileNotFoundError(
            f"Adapter not found at '{path}'. Add the adapter folder to the repo under models/ "
            f"or set the matching environment variable in Render.")
    from peft import PeftModel
    if peft_model is None:
        peft_model = PeftModel.from_pretrained(engine["model"], path, adapter_name=name)
        peft_model.eval()
        engine["peft"] = peft_model
    elif name not in peft_model.peft_config:
        peft_model.load_adapter(path, adapter_name=name)
    peft_model.set_adapter(name)
    return peft_model, nullcontext()


def generate(choice, quant_choice, prompt, max_tokens, sampling, temp, p, k, system_text):
    engine = load_engine(quant_choice)
    tok = engine["tok"]
    messages = []
    if system_text.strip():
        messages.append({"role": "system", "content": system_text.strip()})
    messages.append({"role": "user", "content": prompt.strip()})
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with engine["lock"]:  # one generation at a time: adapter switching is global state
        model, ctx = select_model(engine, choice)
        inputs = tok(text, return_tensors="pt").to(model.device)
        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": sampling,
                      "pad_token_id": tok.eos_token_id}
        if sampling:
            gen_kwargs.update({"temperature": max(0.01, temp), "top_p": p, "top_k": k})
        start = time.time()
        with ctx, torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - start
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return answer, elapsed


tab_chat, tab_compare, tab_details = st.tabs(["Chat / Playground", "Compare models", "Experiment details"])

with tab_chat:
    col1, col2 = st.columns([2, 1])
    with col1:
        prompt = st.text_area("User prompt", height=140,
                              placeholder="I've been anxious about exams and can't concentrate.")
        system_prompt = st.text_area("Optional system instruction", value=SYSTEM_DEFAULT, height=100)
    with col2:
        st.markdown("#### Active settings")
        st.write(f"**Configuration:** {model_choice}")
        st.write(f"**Quantization:** {quantization}")
        st.write(f"**Decoding:** {'Sampling' if do_sample else 'Greedy'}")
        st.write(f"**Max tokens:** {max_new_tokens}")
        st.write(f"**Temperature:** {temperature if do_sample else 'Ignored (greedy)'}")
        st.write(f"**Top-p / Top-k:** {f'{top_p} / {top_k}' if do_sample else 'Ignored (greedy)'}")
    if st.button("Generate response", type="primary"):
        if not prompt.strip():
            st.info("Enter a prompt first.")
        else:
            try:
                with st.spinner("Generating response... (CPU servers can take a minute or more)"):
                    answer, elapsed = generate(model_choice, quantization, prompt, max_new_tokens,
                                               do_sample, temperature, top_p, top_k, system_prompt)
                st.markdown("### Model response")
                st.write(answer)
                st.caption(f"Generation time: {elapsed:.1f} seconds | Configuration: {model_choice}")
                st.caption("Outputs may be inaccurate or incomplete. This prototype is not clinically validated.")
            except Exception as exc:
                st.error(f"Generation failed: {exc}")

with tab_compare:
    st.subheader("Same-prompt comparison")
    compare_prompt = st.text_area("Prompt for all configurations", key="compare_prompt", height=100)
    compare_tokens = st.slider("Comparison max new tokens", 32, 256, 96, 16)
    if st.button("Run comparison"):
        if not compare_prompt.strip():
            st.info("Enter a prompt.")
        else:
            for name in CONFIGS:
                st.markdown(f"### {name}")
                try:
                    answer, elapsed = generate(name, quantization, compare_prompt, compare_tokens, False,
                                               0.7, 0.9, 50,
                                               "You are supportive, practical, and non-diagnostic.")
                    st.write(answer)
                    st.caption(f"{elapsed:.1f} seconds")
                    gc.collect()
                except Exception as exc:
                    st.error(f"{name}: {exc}")

with tab_details:
    st.header("Model and experiment details")
    st.markdown("""
**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`

**Configurations**
- **Base:** unmodified instruction model.
- **SFT (QLoRA):** base plus the saved QLoRA SFT adapter.
- **SFT → ORPO:** final checkpoint from ORPO training initialized from SFT, loaded directly onto the base model.

**Recorded SFT settings:** 100 provisional examples; NF4 + double quantization; LoRA r=8, alpha=16, dropout=0.05; 2 epochs; LR 2e-4; max length 512.

**Recorded ORPO settings:** 20 preference pairs, 1 epoch, LR 5e-6, beta 0.1. The preference data is tiny and reuses a rejected response, so results are preliminary.

**Important:** sidebar controls change generation settings only; they never retrain the model.
""")
    st.info("For self-harm or immediate danger, encourage immediate human assistance and appropriate local "
            "emergency resources. Do not rely on this prototype for crisis response.")
