#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install transformers[torch,sentencepiece] -U bitsandbytes wikipedia langchain langchain-community langchain-huggingface faiss-cpu jq > install_log.txt && powershell -Command "Get-Content install_log.txt | Select-Object -Last 1"


# In[2]:


import torch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# 埋め込みベクトルを作成する埋め込みモデル名を指定
embedding_model_name = "BAAI/bge-m3"

# モデル名から埋め込みモデルを初期化
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"model_kwargs": {"torch_dtype": torch.float16}},
)


# In[3]:


# 訓練データの取得
import wikipedia
wikipedia.set_lang('ja')
page = wikipedia.page('百人町',auto_suggest=False)
content = page.content
content = content+"""
オモドックスクールの住所は新宿区百人町2-4-5-607です。"""
print(len(content))
print(content)


# In[4]:


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 指定数で分割するためのText Splitterを初期化
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,         # 分割時の最大文字数
    chunk_overlap=100,      # 分割文書間で重複させる最大文字数
    add_start_index=True,  # 元文書における開始位置を付与
)

# 文字列を分割用のDocumentオブジェクトに変換
documents = [Document(page_content=content)] 

# 分割の実行
split_documents = text_splitter.split_documents(documents)

# 文書の分割数
print(len(split_documents))
# 先頭文書の文字数
print(len(split_documents[0].page_content))


# In[5]:


import os
from langchain_community.vectorstores import FAISS

# ベクトルストアのパス設定
index_path = "./faiss_index"

# ディレクトリ存在確認 → 読み込み/新規作成
if os.path.exists(index_path):
    vectorstore = FAISS.load_local(
        folder_path=index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    print(f"既存のインデックスを読み込みました: {index_path}")
else:
    vectorstore = FAISS.from_documents(split_documents, embedding_model)
    vectorstore.save_local(index_path)
    print(f"新しいインデックスを作成しました: {index_path}")


# In[6]:


# ベクトルストア内を文書検索するRetrieverを初期化する
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# In[7]:


# 文書の検索を実行
retrieved_documents = retriever.invoke("オモドックスクールの住所は？")

# 検索された文書を確認
print(retrieved_documents)


# In[8]:


# モデルとトークナイザーの読み込み
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーの読み込み
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")


# In[9]:


from transformers import pipeline 
from langchain_huggingface import HuggingFacePipeline

# テキスト生成用のパラメータを指定
generation_config = {
    "max_new_tokens": 30,
    "do_sample": True,      #ランダム回答可
    "temperature": 0.3,     #上記True時の温度パラメータ0.3-1.0
    "top_p": 0.3,           #生成の多様性
}

# テキスト生成を行うパイプラインを作成
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    **generation_config,
)

# パイプラインからLangchainのLLMコンポーネントを作成
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)


# In[10]:


# テンプレート定義
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_template("""
以下の文脈から、質問に簡潔に答えてください。
同じ内容を繰り返さず、余計な記号やフォーマットは使わず、30字以内でまとめてください。

【文脈】
{context}

【質問】
{question}

【回答】""")


# In[11]:


# チェーン構築
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt 
    | llm
)


# In[12]:


# RAGの実行
response = rag_chain.invoke("百人町はどんな街？")
answer = response.split("【回答】")[-1]
print(answer)


# In[13]:


def ai_chat_bot(prompt):
    # RAGの実行
    response = rag_chain.invoke(prompt)
    answer = response.split("【回答】")[-1]
    return answer


# In[23]:


# AIチャットボットの送受信テスト
if __name__ == '__main__':
    prompt = input("プロンプトを入力してください: ")
    print("AIチャットボットからの出力: ", ai_chat_bot(prompt))


# In[ ]:




