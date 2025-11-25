"""
与 OpenAI API 交互的轻量封装。

本文件提供了一个 [GPT_response](file://e:/rumors-in-multi-agent/LLM.py#L22-L92) 函数，用于向指定模型发送 chat-style 的 `messages` 列表并返回模型回复。
注：此模块依赖 `openai` 包（从中导入 `OpenAI` 客户端）和 `tiktoken` 用于估算 token 数量。
"""
import openai
from openai import OpenAI
import tiktoken
import time

# 使用 tiktoken 初始化编码器：
# - [base_enc](file://e:\rumors-in-multi-agent\LLM.py#L14-L14)：通用回退编码（cl100k_base），用于在无法为传入模型获取编码器时回退；
# - [default_enc](file://e:\rumors-in-multi-agent\LLM.py#L16-L16)：默认使用 gpt-4 的编码规则作为参考（保持与旧版行为兼容）。
base_enc = tiktoken.get_encoding("cl100k_base")
assert base_enc.decode(base_enc.encode("hello world")) == "hello world"
default_enc = tiktoken.encoding_for_model("gpt-4")

# 请在实际使用时把这里替换为你的 API Key（或将其置于环境变量中并修改构造逻辑）。
import os
openai_api_key_name = os.getenv("OPENAI_API_KEY", "PUT-YOUR-API-KEY-HERE")

# 添加DeepSeek API的基础URL
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def GPT_response(messages, model_name):
  """
  向 OpenAI chat completion API 发送消息并返回回复文本及 token 计数。

  参数:
  - messages: 列表，每个元素为字典，包含至少键 `content`（字符串），遵循 chat API 的消息格式。
  - model_name: 要调用的模型名称字符串（例如 'gpt-4'）。

  返回:
  - 成功时返回 (reply_text, token_num_count)
  - 在 API 错误路径下可能返回 ('API error', error_message, token_num_count)

  备注:
  - 函数会先估算输入消息使用的 token 数量，然后在收到回复后把回复的 token 数加上并返回。
  - 对短暂失败采用了最多两次重试 + 第三次等待 60 秒再重试的简单策略。
  - 若传入不被支持的模型名，会抛出 ValueError。
  """

  # 尝试为传入的模型获取对应的编码器，用于更准确的 token 计数。
  # 若失败（例如传入一个 tiktoken 不认识的第三方模型名 like 'deepseek'），则回退到 base_enc。
  try:
    local_enc = tiktoken.encoding_for_model(model_name)
  except Exception:
    # 如果无法为指定模型获取编码器，使用通用回退编码并打印提示（不会中断调用）。
    print(f"Warning: 无法为模型 '{model_name}' 获取 tiktoken 编码器，使用 cl100k_base 回退进行估算。")
    local_enc = base_enc

  # 统计输入 messages 中的 token 数量（仅对 message.content 做了统计）
  token_num_count = 0
  for item in messages:
    token_num_count += len(local_enc.encode(item.get("content", "")))

  # 根据模型名称确定API基础URL
  base_url = DEEPSEEK_BASE_URL if 'deepseek' in model_name.lower() else None

  # 创建 OpenAI 客户端；api_key 参数可以改为从环境变量读取以提高安全性
  client = OpenAI(
      # defaults to os.environ.get("OPENAI_API_KEY")
      api_key=openai_api_key_name,
      base_url=base_url
  )

  # 更稳健的重试逻辑：最多三次尝试（0,1,2），在第二次与第三次之间做 60 秒等待。
  result = None
  for attempt in range(3):
    try:
      result = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
      break
    except Exception as e:
      # 当达到最后一次尝试仍失败时，返回错误信息和当前 token 估算
      if attempt == 2:
        print(f"OpenAI API error after retries: {str(e)}")
        return 'API error', str(e), token_num_count
      # 在第二次尝试之前进行短等待以减轻速率/临时网络问题
      if attempt == 1:
        print(f"{model_name} Waiting 60 seconds for API query (attempt {attempt + 1})")
        time.sleep(60)
      else:
        # 立刻重试（第一次失败后的即时重试）
        continue

  # 将模型回复的 token 数也计入总数（用于记录/计费测算）
  # 使用 local_enc 来估算回复长度（若模型特定编码不可用，已回退到 base_enc）
  reply_text = result.choices[0].message.content
  token_num_count += len(local_enc.encode(reply_text))
  print(f'Token_num_count: {token_num_count}')
  # 返回文本内容和 token 总数
  return reply_text, token_num_count