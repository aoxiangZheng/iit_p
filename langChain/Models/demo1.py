from langchain.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

llm = Tongyi(
    model_name="qwen-plus",
    temperature=0.7
)

# prompt1
prompt1 = PromptTemplate(
    input_variables=["product"],
    template="为 {product} 写一个简短描述："
)
# prompt2
prompt2 = PromptTemplate(
    input_variables=["description"],
    template="这个产品的合理价格是多少？\n描述：{description}"
)

# 新链式写法
description_chain = prompt1 | llm
price_chain = prompt2 | llm
# 先生成描述
desc_result = description_chain.invoke({"product": "无线耳机"})
# 再用描述生成价格
price_result = price_chain.invoke({"description": desc_result})
print("描述:", desc_result)
print("价格:", price_result)