from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化通义千问模型（使用 OpenAI 兼容模式）
llm = OpenAI(
    model_name="qwen-max",  # 千问模型名
    temperature=0.7,        # 控制随机性（0-1）
    max_tokens=200,         # 最大生成token数
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),  # 使用环境变量中的API密钥
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 千问OpenAI兼容地址
)

# 示例1：直接调用模型
result = llm("你好，请介绍一下你自己")
print("示例1 - 直接调用模型:")
print(result)
print("-" * 50) 