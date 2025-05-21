# app.py
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


# 加载模型和分词器
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-shakespeare")
    model = GPT2LMHeadModel.from_pretrained("./gpt2-shakespeare")
    model.eval()
    return tokenizer, model


tokenizer, model = load_model()

# 网页标题
st.title("🎭 莎士比亚风格聊天机器人")

# 聊天历史记录
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 用户输入
user_input = st.text_input("你想对莎士比亚说点什么？", key="user_input")

# 响应生成逻辑
if user_input:
    st.session_state.chat_history.append(("你", user_input))

    # 添加 prompt 提示
    prompt = f"User: {user_input}\nShakespeare:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=150,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # 提取回复部分
    response = output_text[len(prompt):].strip().split("\n")[0]
    st.session_state.chat_history.append(("莎士比亚", response))


# 显示聊天记录
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")
