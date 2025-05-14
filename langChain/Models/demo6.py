from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化通义千问聊天模型
llm = ChatOpenAI(
    model="qwen-max",
    temperature=0.7,
    max_tokens=200,
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 加载工具
tools = load_tools(["llm-math"], llm=llm)

# 初始化 Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 示例6：使用 Agent 进行数学计算
print("示例6 - 使用 Agent 进行数学计算:")
response = agent.run("计算 123 的平方根是多少？")
print(response)
print("-" * 50) 