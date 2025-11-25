# hybrid_run.py
import json
import numpy as np
from LLM import GPT_response
from hybrid_prompt import generate_hybrid_prompt
import networkx as nx

# 状态定义
STATE_S = 0
STATE_I = 1 # Rumor
STATE_A = 2 # Anti
STATE_M = 3 # Motivation
STATE_R = 4 # Recovered

def update_agent_state(current_state, action):
    """简单状态机"""
    if action == 'SPREAD': return STATE_I
    if action == 'ANTI': return STATE_A
    if action == 'MOTIVATE': return STATE_M
    if action == 'IGNORE': return current_state # 保持原样
    return current_state

def parse_hybrid_response(text):
    # 简单的解析逻辑，需配合正则增强鲁棒性
    lines = text.strip().split('\n')
    action = 'IGNORE'
    content = ''
    check_vals = []
    
    for line in lines:
        if line.startswith('ACTION:'):
            action = line.split(':', 1)[1].strip()
        elif line.startswith('CONTENT:'):
            content = line.split(':', 1)[1].strip()
        elif line.strip() in ['True', 'False']:
            check_vals.append(line.strip() == 'True')
            
    return action, content, check_vals

def run_hybrid_simulation(saving_path, G, agent_config_path, steps=50):
    # 1. 初始化
    with open(agent_config_path, 'r') as f: # 读取包含 Per/Pos 等参数的配置
        # 假设这是个 list 或 dict，这里简化处理
        agents_data = json.load(f) 
        
    num_agents = len(agents_data)
    agent_states = [STATE_S] * num_agents
    post_histories = [""] * num_agents
    
    # 初始谣言投放 (Patient Zero)
    seed_agent = 0
    agent_states[seed_agent] = STATE_I
    post_histories[seed_agent] = "System: Rumor started here."
    
    # 宏观参数
    global_pop = 1.0 # 初始热度
    decay_rate = 0.95
    
    results = []
    
    for t in range(steps):
        print(f"--- Step {t} ---")
        # 更新热度 Wang (2025) Eq.6
        global_pop *= decay_rate 
        
        # 激活 Agent (基于 Pos 加权)
        # active_indices = select_active_agents(agents_data) 
        # 简化：随机选 5 个
        active_indices = np.random.choice(num_agents, 5, replace=False)
        
        for idx in active_indices:
            agent = agents_data[str(idx)] # 假设 ID 是字符串 key
            
            # 获取邻居状态
            neighbors = list(G.neighbors(int(idx)))
            n_statuses = [agent_states[n] for n in neighbors]
            
            # 生成 Prompt
            prompt = generate_hybrid_prompt(
                agent, n_statuses, global_pop, 
                post_histories[idx], ["Rumor A"], []
            )
            
            # LLM 调用
            messages = [{"role": "user", "content": prompt}]
            resp, _ = GPT_response(messages, "gpt-4o-mini") # 或 deepseek
            
            # 解析与更新
            action, content, checks = parse_hybrid_response(resp)
            new_state = update_agent_state(agent_states[idx], action)
            agent_states[idx] = new_state
            
            # 传播内容 (Hu 的机制)
            if content:
                post_histories[idx] += f"Me: {content}\n"
                for n in neighbors:
                    post_histories[n] += f"{agent['agent_name']}: {content}\n"
                    
        # 记录数据
        state_counts = [agent_states.count(s) for s in range(5)]
        results.append(state_counts)
        print(f"States: S={state_counts[0]}, I={state_counts[1]}, A={state_counts[2]}")
        
    return results