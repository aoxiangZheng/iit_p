from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 示例7：使用 Document Loaders 加载和处理文档
print("示例7 - 使用 Document Loaders 加载和处理文档:")

# 创建示例文本文件
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("""
    人工智能（AI）是计算机科学的一个分支，它致力于创造能够模拟人类智能的系统。
    机器学习是AI的一个子领域，它使用统计方法让计算机系统能够从数据中学习。
    深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。
    """)

# 加载文档
loader = TextLoader("example.txt")
documents = loader.load()

# 分割文档
text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
texts = text_splitter.split_documents(documents)

# 创建向量存储
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
db = Chroma.from_documents(texts, embeddings)

# 搜索相似内容
query = "什么是机器学习？"
docs = db.similarity_search(query)
print(f"\n查询: {query}")
print(f"找到的相关内容: {docs[0].page_content}")

# 清理临时文件
os.remove("example.txt")
print("-" * 50) 