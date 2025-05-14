from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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

# 示例3：使用 LLMChain
prompt = PromptTemplate(
    input_variables=["topic"],
    template="请用中文回答：{topic}"
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("请用中文回答：1+1等于几？")
print("示例3 - 使用 LLMChain:")
print(result)
print("-" * 50) 