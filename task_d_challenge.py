"""
任务 D：综合项目 — 完整的农业 AI 助手 Web 应用
- 整合 RAG 系统和 agicto 云端 API
- 在 Docker 容器中运行
- Streamlit Web UI
"""

import os
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ── 页面配置 ──
st.set_page_config(page_title="农业 AI 助手", page_icon="🌿", layout="wide")
st.title("农业 AI 知识问答系统")
st.caption("基于 RAG 技术，整合农业知识库与大语言模型")

# ── 侧边栏：配置 ──
st.sidebar.header("系统配置")

api_key = st.sidebar.text_input(
    "API Key", type="password", value=os.environ.get("AGICTO_API_KEY", "")
)
model_name = st.sidebar.selectbox("选择模型", ["qwen-plus", "gpt-4o-mini"])
top_k = st.sidebar.slider("检索文档数 (top_k)", 1, 5, 3)

base_url = "https://api.agicto.cn/v1"

# ── 初始化客户端 ──


@st.cache_resource
def init_clients(_api_key, _base_url):
    """初始化 LLM 客户端和向量数据库"""
    llm_client = OpenAI(api_key=_api_key, base_url=_base_url)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3"
    )
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = chroma_client.get_collection(
            name="agri_knowledge", embedding_function=embedder
        )
    except Exception:
        collection = None
    return llm_client, collection


llm_client, collection = init_clients(api_key, base_url)

if collection is None:
    st.sidebar.warning("知识库未初始化，请先运行 Task B 构建知识库")

# ── 检索函数 ──


def retrieve_knowledge(question, top_k=3):
    """从知识库中检索相关知识"""
    if collection is None:
        return []
    results = collection.query(query_texts=[question], n_results=top_k)
    return list(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    )


# ── 问答函数 ──
def ask_with_rag(question, top_k=3):
    """RAG 问答"""
    # 检索
    knowledge = retrieve_knowledge(question, top_k)

    if not knowledge:
        # 无知识库，直接回答
        messages = [
            {"role": "system", "content": "你是农业专家。"},
            {"role": "user", "content": question},
        ]
        response = llm_client.chat.completions.create(
            model=model_name, messages=messages, temperature=0.5, max_tokens=1024
        )
        return response.choices[0].message.content, []

    # 构建上下文
    context_parts = []
    for doc, meta, dist in knowledge:
        context_parts.append(f"来源: {meta['source']}\n内容: {doc}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """你是农业植保专家。请基于以下参考资料回答用户的问题。
要求：
1. 回答必须基于参考资料
2. 如果资料不足以回答问题，请如实告知
3. 回答要实用、有针对性"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"参考资料:\n{context}\n\n问题: {question}",
        },
    ]

    response = llm_client.chat.completions.create(
        model=model_name, messages=messages, temperature=0.3, max_tokens=1024
    )

    return response.choices[0].message.content, knowledge


# ── 主界面 ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
question = st.chat_input("请输入你的农业问题...")

if question:
    # 显示用户问题
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            answer, knowledge = ask_with_rag(question, top_k)
            st.markdown(answer)

    # 显示参考来源
    if knowledge:
        with st.expander("参考来源"):
            for i, (doc, meta, dist) in enumerate(knowledge):
                st.markdown(
                    f"**来源 {i + 1}**: {meta['source']} (相关度: {1 - dist:.2f})"
                )
                st.text(doc[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── 知识库预览 ──
with st.sidebar.expander("知识库文档列表"):
    if collection:
        count = collection.count()
        st.write(f"共 {count} 个文本块")
    else:
        st.write("知识库未初始化")
