import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
prompt = PromptTemplate(
    input_variables=["expert", "question"],
    template="{expert}として、次の質問に答えてください: {question}"
)

chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.title("LangChain × Streamlit サンプルアプリ: 専門家QAチャット")
st.write(
    "以下の入力欄に質問や相談内容を入力し、モード（専門家の種類）を選択して「送信」ボタンを押してください。"
)

# ラジオボタンで専門家モードを選択
expert = st.radio(
    "専門家モードを選択してください:",
    ("金融専門家", "医療専門家", "汎用コンサルタント")
)

# テキスト入力フォーム
user_input = st.text_area(
    label="入力テキスト",
    height=150,
    placeholder="ここに質問や相談内容を入力してください。"
)

# レスポンス生成関数
def generate_response(question: str, expert_mode: str) -> str:
    """
    LangChain の LLMChain を使って回答を生成する
    """
    return chain.run(expert=expert_mode, question=question)

# 送信ボタン押下時の処理
if st.button("送信"):
    if user_input.strip():
        with st.spinner("回答を生成中..."):
            try:
                answer = generate_response(user_input, expert)
                st.subheader("回答結果")
                st.write(answer)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
    else:
        st.error("質問内容を入力してから送信してください。")
