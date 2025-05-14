from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
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

# 创建对话记忆
memory = ConversationBufferMemory()

# 创建对话链
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose=True
)

# 示例5：使用 Memory 进行带记忆的对话
print("示例5 - 使用 Memory 进行带记忆的对话:")
print("第一轮对话:")
response1 = conversation.predict(input="你好，我叫小明")
print(response1)
print("\n第二轮对话:")
response2 = conversation.predict(input="你还记得我的名字吗？")
print(response2)
print("-" * 50) 