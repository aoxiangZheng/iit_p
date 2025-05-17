from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

# 配置 Azure OpenAI
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = "gpt-4"
api_version = "2025-01-01-preview"

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    deployment_name=deployment_name,
    api_version=api_version,
    temperature=0,
)

# 工具定义（保持不变）
@tool
def query_author(name: str) -> str:
    """根据文章名查询作者信息"""
    print(f"[query_author] 输入: {name}")
    prompt = PromptTemplate(
        input_variables=["name"],
        template="请告诉我《{name}》的作者是谁？"
    )
    return llm.invoke(prompt.format(name=name)).content or "未查到作者"

@tool
def query_article(title: str) -> str:
    """根据文章名查询原文"""
    print(f"[query_article] 输入: {title}")
    prompt = PromptTemplate(
        input_variables=["title"],
        template="请给出《{title}》的原文内容。"
    )
    return llm.invoke(prompt.format(title=title)).content

@tool
def analyze_background_and_roles(article: str) -> str:
    """分析文章背景和角色"""
    print(f"[analyze_background_and_roles] 输入: {article}")
    prompt = PromptTemplate(
        input_variables=["article"],
        template="请根据以下文章内容，分析其时代背景、环境背景和主要角色。要求分条详细说明。\n\n文章内容：\n{article}"
    )
    return llm.invoke(prompt.format(article=article)).content

@tool
def plot_summary(article: str) -> str:
    """影视剧本编剧"""
    print(f"[plot_summary] 输入: {article}")
    prompt = PromptTemplate(
        input_variables=["article"],
        template="你是一位专业的影视编剧。请将下面的文章内容，提炼细腻的剧情梗概，要求包括主要事件、关键角色、场景变化和情感转折。请用现代白话文分条列出，便于后续分镜设计。\n\n文章内容：\n{article}"
    )
    return llm.invoke(prompt.format(article=article)).content

@tool
def storyboard(plot: str) -> str:
    """影视分镜师"""
    print(f"[storyboard] 输入: {plot}")
    prompt = PromptTemplate(
        input_variables=["plot"],
        template="你是一位专业的影视分镜师。请根据以下剧情梗概，为每个事件设计详细的分镜脚本。每个分镜请包含：场景描述、角色与动作、主要对白、镜头运动方式、镜头时长建议（秒）。请用结构化JSON格式输出。\n\n剧情梗概：\n{plot}"
    )
    return llm.invoke(prompt.format(plot=plot)).content

# 优化提示
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一个智能助手，任务是根据用户输入的文章名，完成以下步骤："
        "1. 查询文章原文；2. 分析时代背景、环境背景和主要角色；3. 生成剧情梗概；4. 制作分镜脚本。"
        "请按顺序调用工具，确保每步完成后将结果传递给下一步。"
    )),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

tools = [query_author, query_article, analyze_background_and_roles, plot_summary, storyboard]
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 测试
result = agent_executor.invoke({"input": "春晓"})
print(result)