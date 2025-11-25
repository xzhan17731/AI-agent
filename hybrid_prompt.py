# hybrid_prompt.py
from prompt_rumor_test import likely_to_accept_rumors, likely_to_forward_rumors

def calculate_payoff(agent, neighbors_status, global_pop):
    """
    根据 Wang (2025) 公式计算收益:
    Pro_R = (1 + epsilon1) * P1 * Inf_r
    Pro_A = (1 + epsilon2) * P2 * Inf_ar
    """
    # 简化版参数
    epsilon1 = 0.2 # 动力谣言对谣言的增益
    epsilon2 = 0.1 # 动力谣言对反谣言的增益
    
    # 统计邻居状态比例 (0:S, 1:I-Rumor, 2:A-Anti, 3:M-Motivation)
    total = len(neighbors_status) if neighbors_status else 1
    p1 = neighbors_status.count(1) / total # Rumor
    p2 = neighbors_status.count(2) / total # Anti
    p3 = neighbors_status.count(3) / total # Motivation
    
    # 结合 Agent 自身兴趣 (Mat) 和活跃度 (Pos) 作为影响力因子
    inf_base = agent['Mat'] * agent['Pos'] * global_pop
    
    payoff_r = (1 + epsilon1) * p1 * inf_base
    payoff_a = (1 + epsilon2) * p2 * inf_base
    
    return payoff_r, payoff_a, p1, p2, p3

def generate_hybrid_prompt(agent, neighbors_status, global_pop, post_history, rumor_list, current_belief):
    """
    生成包含博弈情境的 Prompt
    """
    payoff_r, payoff_a, p1, p2, p3 = calculate_payoff(agent, neighbors_status, global_pop)
    
    persona_desc = f"You have a Information Perception of {agent['Per']:.2f} and Activity level of {agent['Pos']:.2f}. "
    if agent['Cog'] > 0.7:
        persona_desc += "You are highly rational and skeptical of unverified info. "
    else:
        persona_desc += "You are easily influenced by emotional content. "
        
    context_desc = f"""
    [Social Context & Game Theory]
    - {p1*100:.1f}% of your neighbors are spreading the rumor.
    - {p2*100:.1f}% are refuting it (Anti-Rumor).
    - {p3*100:.1f}% are posting emotional support/motivation about it.
    
    Your calculated social payoff for spreading the rumor is {payoff_r:.2f}, and for refuting is {payoff_a:.2f}.
    If Payoff_Rumor > Payoff_Anti, you feel social pressure to join the spread.
    """
    
    base_prompt = f"""
    Hi {agent['agent_name']}, {persona_desc}
    You are active on a social network.
    {context_desc}
    
    Read the post history: {{{post_history}}}
    
    Based on your Persona and Social Payoff, decide your action.
    You can choose to:
    1. [SPREAD]: Spread the rumor directly.
    2. [ANTI]: Refute the rumor (Anti-Rumor).
    3. [MOTIVATE]: Share emotional/煽动性 content supporting the rumor (Motivation-Rumor).
    4. [IGNORE]: Do nothing.
    
    [Output Format]
    ACTION: [SPREAD / ANTI / MOTIVATE / IGNORE]
    CONTENT: [Your post content]
    CHECK:
    [True/False] for each rumor in list: {rumor_list}
    """
    return base_prompt