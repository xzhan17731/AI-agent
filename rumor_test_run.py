"""
主运行脚本：模拟多智能体在社交网络环境中传播谣言/帖子的实验驱动脚本。

主要职责：
- 读取保存的 agent 配置、谣言与帖子列表。
- 根据给定策略（随机/按好友数优先）选择行动 agent，并使用 LLM 生成其发帖内容。
- 对模型返回进行解析（POST / CHECK），更新 agent 的历史及其朋友的历史，维护谣言相信/否定矩阵。

注意：此脚本期望目录 `Saving_path` 下存在按 `agent_{i}` 结构保存的 agent json 文件，以及 `rumor_list.json` 与 `posts_list.json`。
"""

from LLM import *
from rumor_test_env import *
from prompt_rumor_test import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
import sys

# 确保输出使用 UTF-8，以避免 Windows 控制台下的编码错误
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'


def safe_print(*args, **kwargs):
    """
    一个对 print 的安全封装，处理可能的 Unicode 编码错误。

    在某些 Windows 环境或日志系统中，直接 print 含有特殊字符的字符串可能抛出
    UnicodeEncodeError；此函数会在遇到该异常时将每个参数进行 utf-8 编码替换后再打印，
    以保证不会因单条输出导致程序崩溃。
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # 将无法编码的字符用替代字符替换，保证可打印性
        encoded_args = [str(arg).encode('utf-8', errors='replace').decode('utf-8') for arg in args]
        print(*encoded_args, **kwargs)


def analyze_input(text):
    """
    解析 LLM 返回的文本，提取 POST 和 CHECK 两个部分。

    期望的输入格式（示例）:
    POST some content ... CHECK True\nFalse\nTrue\nFalse

    返回:
    - post: POST 部分的文本内容（字符串）
    - check_list: 长度至少为 4 的布尔列表，表示对每条谣言的判断（True/False）

    实现细节：使用正则分别匹配 POST 和 CHECK 区段，CHECK 下的每一行必须以 True 或 False 开头，
    否则抛出 ValueError。若解析到的 check 列表长度不足 4，会用 False 填充（临时方案）。
    """
    # 使用 DOTALL 使 '.' 匹配换行，以便捕获多行 POST 内容
    post_pattern = re.compile(r'POST\s+(.*?)(?=\s+CHECK)', re.DOTALL)
    check_pattern = re.compile(r'CHECK\s+(.*)', re.DOTALL)

    # 查找 POST 部分
    post_match = post_pattern.search(text)
    if not post_match:
        raise ValueError("POST section not found or incorrectly formatted.")
    post = post_match.group(1).strip()
    if not post:
        raise ValueError("POST content is empty.")

    # 查找 CHECK 部分
    check_match = check_pattern.search(text)
    if not check_match:
        raise ValueError("CHECK section not found or incorrectly formatted.")

    # 将 CHECK 部分按行分割并解析每一行的布尔值（要求以 True 或 False 开头）
    check_lines = check_match.group(1).strip().split('\n')
    check_list = []
    for line in check_lines:
        if not re.match(r'^(True|False)', line):
            raise ValueError("Lines after CHECK must start with True or False.")
        # 以 'True' 开头则视为 True，否则为 False
        check_list.append(line.startswith('True'))

    # 临时兼容：如果返回的判断数量不足 4，则补齐为 False，保证后续索引安全
    while len(check_list) < 4:
        check_list.append(False)

    return post, check_list


def run_exp(Saving_path, iteration_num, query_time_limit, agent_count, num_of_initial_posts, dialogue_history_method='all_history', selection_policy = 'random', patient_zero_policy = 'random', model_name = 'deepseek-chat'):
    """
    运行一次完整的实验。

    参数:
    - Saving_path: 存放实验数据（agent 配置、谣言、帖子等）的目录路径
    - iteration_num: 实验轮次序号（用于记录/日志）
    - query_time_limit: 时间步数（每个时间步会有一次 agent 行动）
    - agent_count: agent 总数
    - num_of_initial_posts: 每个 agent 初始化时包含的随机帖子数量
    - dialogue_history_method: 传入给 prompt 的历史对话策略
    - selection_policy: 选择行动 agent 的策略（'random' 或 'more_friend_first'）
    - patient_zero_policy: 初始谣言分配策略（'random' 或 'more_friend_first'）
    - model_name: 调用的 LLM 模型名称

    返回:
    - rumor_matrix: 最终的谣言相信矩阵（agent_count x len(rumor_list)）
    """

    agent_dir = Saving_path
    agent_list = []

    # 读取谣言与帖子列表（JSON 文件）
    with open(Saving_path+f'/rumor_list.json', 'r') as f:
        rumor_list = json.load(f)
    with open(Saving_path+f'/posts_list.json', 'r') as f:
        posts_list = json.load(f)

    # 读取每个 agent 的配置
    for i in range(agent_count):
        with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'r') as f:
            agent_list.append(json.load(f))

    # 初始化谣言矩阵（全部为 False），并初始化每个 agent 的历史帖子与计数器
    rumor_matrix = [[False for _ in range(len(rumor_list))] for __ in range(agent_count)]
    safe_print(rumor_matrix)
    post_history = ['' for _ in range(agent_count)]
    post_count = [0 for _ in range(agent_count)]

    # 将谣言随机分配（或根据策略）给某些 agent，作为 patient zero
    rumor_list_copy = rumor_list.copy()

    if patient_zero_policy == 'random':
        while rumor_list_copy:
            random_agent = random.randint(0, agent_count-1)
            random_key = random.choice(list(rumor_list_copy.keys()))
            random_rumor = rumor_list_copy.pop(random_key)
            post_history[random_agent] += f'Random post: {random_rumor}\n'
            post_count[random_agent] += 1

            safe_print(f'Rumor {random_key}: {random_rumor} is assigned to Agent {random_agent}')

    elif patient_zero_policy == 'more_friend_first':
        # 将谣言优先分配给好友数最多的 agent
        while rumor_list_copy:
            weights = [len(agent['friends']) for agent in agent_list]
            top_agent = weights.index(max(weights))
            random_key = random.choice(list(rumor_list_copy.keys()))
            random_rumor = rumor_list_copy.pop(random_key)
            post_history[top_agent] += f'Random post: {random_rumor}\n'
            post_count[top_agent] += 1
            weights.pop(top_agent)

    # 为每个 agent 补足初始随机帖子，直到达到 num_of_initial_posts
    for i in range(agent_count):
        while post_count[i] < num_of_initial_posts:
            random_key = random.choice(list(posts_list.keys()))
            post_history[i] += f'Random post: {posts_list[random_key]}\n'
            post_count[i] += 1

        safe_print(f'Initialization for agent {i}, info: {agent_list[i]}\n Initial {post_count[i]} posts: {post_history[i]}')

    safe_print('Initialization done!')

    # 主时间步循环：每个时间步选择一个 agent 执行一次动作（发帖 / 判断）
    for ts in range(query_time_limit):
        safe_print(f'\n===============================================================\n')
        safe_print(f'Timestamp {ts}')

        # 根据策略选择 agent
        if selection_policy == 'random':
            i = random.randint(0, agent_count-1)
        elif selection_policy == 'more_friend_first':
            weights = [len(agent['friends']) for agent in agent_list]
            i = random.choices(range(len(agent_list)), weights=weights, k=1)[0]

        safe_print(f'\nPick agent {i} to act')
        ag = agent_list[i]

        # 1) 使用 prompt 工具生成传递给 LLM 的 prompt（包含 agent 信息与其历史）
        prompt = input_prompt_local_agent_DMAS_dialogue_func(ag['agent_name'], ag['agent_age'], ag['agent_job'], 
                                                                ag['agent_traits'], ag['friends'],
                                                                ag['agent_rumors_acc'], ag['agent_rumors_spread'],
                                                                post_history[i], rumor_list, rumor_matrix[i],
                                                                dialogue_history_method)
        safe_print(f'\nFeeding prompt to ChatGPT: \n{prompt}')

        # 2) 构建 messages 并调用 LLM（封装在 LLM.GPT_response 中）
        messages=[{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": prompt})

        initial_response, token_num_count = GPT_response(messages,model_name)
        safe_print(f'\nGetting response from ChatGPT: \n{initial_response}')

        # 3) 解析 LLM 的返回（POST 与 CHECK 部分）
        post, check_list = analyze_input(initial_response)

        # 3a) 将帖子追加到 agent 自身的历史，并传播到其朋友的历史中
        safe_print(f'\nAppending post: {post} to history')
        post_history[i] += f"{ag['agent_name']}: {post}\n"
        for friend in ag['friends']:
            safe_print(f'Update to friend {friend}')
            post_history[friend] += f"{ag['agent_name']}: {post}\n"

        # 3b) 将 CHECK 的判断结果记录到谣言矩阵中（agent 对每条谣言的相信/否定）
        for ru in range(len(rumor_list)):
            safe_print(f"\nAgent {i} {ag['agent_name']} believes rumor {ru} {rumor_list[str(ru)]} is {check_list[ru]}")
        rumor_matrix[i] = check_list

        # 4) 将本轮的谣言矩阵追加写入文本文件（以便后续分析）
        with open(Saving_path+f'/rumor_matrix.txt', 'a') as file:
            np.savetxt(file, rumor_matrix, fmt='%d')
            file.write('\n\n')

    return rumor_matrix


random.seed(66) # 保持实验可复现性

Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/' # 请将此路径替换为实际的代码目录路径
Saving_path = Code_dir_path + 'Env_Rumor_Test'
model_name = 'gpt-4o-mini-2024-07-18'  #'gpt-4-0613', 'gpt-3.5-turbo-16k-0613' # 可替换模型
safe_print(f'-------------------Model name: {model_name}-------------------')

query_time_limit = 500
iterations = 1
agent_count = 100
num_of_initial_posts = 2

for iteration_num in range(iterations):
    safe_print('-------###-------###-------###-------')
    safe_print(f'Iteration num is: {iteration_num}\n\n')
    # 运行实验并获取最终的谣言相信矩阵
    rumor_matrix = run_exp(Saving_path, iteration_num, query_time_limit, agent_count, num_of_initial_posts, dialogue_history_method='all_history',
            selection_policy = 'random', patient_zero_policy = 'random', model_name = model_name)

    safe_print(f'Done')
    '''
    # 下面为示例的结果写出逻辑（目前被注释），可以按需打开并填充对应变量
    with open(Saving_path_result + '/token_num_count.txt', 'w') as f:
        for token_num_num_count in token_num_count_list:
        f.write(str(token_num_num_count) + '\n')

    with open(Saving_path_result + '/success_failure.txt', 'w') as f:
        f.write(success_failure)

    with open(Saving_path_result + '/env_action_times.txt', 'w') as f:
        f.write(f'{index_query_times+1}')
    safe_print(success_failure)
    safe_print(f'Iteration number: {index_query_times+1}')
    '''

