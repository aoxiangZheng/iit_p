from langchain.prompts import PromptTemplate

# 示例2：使用 PromptTemplate
prompt = PromptTemplate(
    input_variables=["topic"],
    template="请用中文回答：{topic}"
)
result = prompt.format(topic="什么是人工智能？")
print("示例2 - 使用 PromptTemplate:")
print(result)
print("-" * 50) 