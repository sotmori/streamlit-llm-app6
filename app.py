import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# load environment variables from .env
load_dotenv()

# retrieve API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("環境変数 OPENAI_API_KEY が設定されていません。.env ファイルを確認してください。")
    st.stop()

# initialize LLM using core OpenAI (avoiding langchain_community dependency)
llm = OpenAI(
    openai_api_key=api_key,
    model_name="gpt-4o-mini",  # 必要に応じて変更
    temperature=0.0
)

# define prompt template
template = "{expert}として、次の質問に答えてください: {question}"
prompt = PromptTemplate(
    input_variables=["expert", "question"],
    template=template
)
chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.set_page_config(page_title="専門家QAチャット", layout="centered")
st.title("LangChain × Streamlit サンプルアプリ: 専門家QAチャット")
st.write("以下の入力欄に質問や相談内容を入力し、モード（専門家の種類）を選択して「送信」ボタンを押してください。")

# radio for expert mode
expert = st.radio(
    "専門家モードを選択してください:",
    ("金融専門家", "医療専門家", "汎用コンサルタント")
)

# text area for user input
user_input = st.text_area(
    label="入力テキスト",
    height=150,
    placeholder="ここに質問や相談内容を入力してください。"
)

# on button click, generate and show response
if st.button("送信"):
    if not user_input.strip():
        st.error("質問内容を入力してから送信してください。")
    else:
        with st.spinner("回答を生成中..."):
            try:
                result = chain.run(expert=expert, question=user_input)
                st.subheader("回答結果")
                st.write(result)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
