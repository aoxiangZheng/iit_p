from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化通义千问聊天模型
chat = ChatOpenAI(
    model="qwen-max",
    temperature=0.7,
    max_tokens=200,
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 示例4：使用 ChatModels 进行多轮对话
messages = [
    SystemMessage(content="你是一个专业的AI助手，请用中文回答。"),
    HumanMessage(content="你好，请介绍一下你自己"),
    HumanMessage(content="你能帮我做什么？")
]

response = chat(messages)
print("示例4 - 使用 ChatModels 进行多轮对话:")
print(response.content)
print("-" * 50) 