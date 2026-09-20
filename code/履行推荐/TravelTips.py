AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

import requests
import os
import time


def get_weather(city: str) -> str:
    """
    通过调用wttr.in API 查询真实的天气信息
    :param city:
    :return:
    """

    # 定义天气url，获取json格式的天气信息
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发送HTTP GET请求获取天气信息
        response = requests.get(url)

        # 检查请求是否成功200
        response.raise_for_status()

        # json解析
        data = response.json()

        # 解析json数据，获取当前天气条件
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']

        temp_c = current_condition['temp_C']

        return f"{city}的当前天气是{weather_desc}，温度是{temp_c}摄氏度。"

    except requests.exceptions.RequestException as e:
        return f"获取{city}天气信息时网络出错：{e}"

    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气,使用博查API搜索并返回优化后的景点推荐
    :param city:
    :param weather:
    :return:
    """
    anysearch_api_url = "https://api.anysearch.com/v1/search"

    # 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"

    payload = {
        "query": query,
        "max_results": 10
    }

    try:
        response = requests.post(anysearch_api_url, json=payload)
        response.raise_for_status()

        results = response.json()['data']['results']

        if not results:
            return "抱歉，没有找到相关结果。"

        # 将搜索结果格式化为文本返回
        formatted = []
        for r in results:
            title = r.get('title', '无标题')
            snippet = r.get('snippet', '')
            formatted.append(f"{title}\n{snippet}")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"抱歉，请求出错：{e}"


# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction
}

from openai import OpenAI

model = 'GLM-4-Flash'
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise SystemExit("错误: 未设置 ZHIPUAI_API_KEY 环境变量。请先执行: export ZHIPUAI_API_KEY=\"你的key\"")

url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}


def generate(prompt: str, sysPrompt: str) -> str:
    """
    调用ZhipuAI的API生成文本,失败时自动重试
    :param prompt:
    :return:
    """
    print("正在调用大语言模型...")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "system", "content": sysPrompt},
        ]
    }
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            data = response.json()
            if response.status_code == 200 and 'choices' in data:
                print("大语言模型响应成功。")
                return data['choices'][0]['message']['content']
            print(f"大模型API异常(第{attempt + 1}次): HTTP {response.status_code}, 响应: {response.text[:200]}")
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"大模型请求失败(第{attempt + 1}次): {e}")
        time.sleep(2)
    raise RuntimeError("调用大语言模型连续3次失败,请检查网络、API Key或账户额度。")

import re
# --- 2. 初始化 ---
user_city = input("请输入你想查询的城市：").strip()
user_prompt = f"你好，请帮我查询一下今天{user_city}的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]
print(f"用户输入: {user_prompt}\n" + "="*40)

for i in range(5):
    print(f"--- 循环 {i+1} ---\n")
    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)

    # 3.2. 调用LLM进行思考
    llm_output = generate(full_prompt, AGENT_SYSTEM_PROMPT)

    # 模型可能会输出多余的Thought-Action，需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)
        if final_match:
            final_answer = final_match.group(1)
            print(f"任务完成，最终答案: {final_answer}")
            break
        observation = "错误: Finish 必须使用 Finish[最终答案] 格式，最终答案需放在方括号内。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue

    # Action 必须是工具调用格式,否则反馈错误让模型重试
    tool_match = re.search(r"(\w+)\(", action_str)
    if not tool_match:
        observation = "错误: Action 必须是工具调用格式 function_name(arg=\"value\") 或 Finish[最终答案]，不能是普通文本。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    tool_name = tool_match.group(1)
    args_str = re.search(r"\((.*)\)", action_str, re.DOTALL).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)