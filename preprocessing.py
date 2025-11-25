# preprocessing.py
import json
import networkx as nx
import numpy as np
from sklearn.neighbors import KDTree
import os

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def calculate_agent_params(agent_data, G):
    """
    计算 Wang 等 (2025) 定义的 Agent 静态属性：
    Per (信息感知度): 基于度数/邻居数
    Pos (活跃度): 基于历史发帖意愿 (agent_rumors_spread)
    Mat (兴趣匹配度): 随机初始化或基于 Persona 关键词
    Cog (认知调节因子): 基于 rumors_acc (对谣言的接受度，越高 Cog 越低)
    """
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    
    params = {}
    for node_id, agent in agent_data.items():
        node_id = int(node_id)
        # Per: 归一化度数
        per = degrees.get(node_id, 0) / max_degree
        
        # Pos: 映射 spread (1-3) 到 0-1
        pos = int(agent['agent_rumors_spread']) / 3.0
        
        # Cog: 映射 acc (1-4) 到 0-1 (acc越高，鉴别力Cog越低)
        # acc=1 -> Cog=1.0; acc=4 -> Cog=0.25
        acc = int(agent['agent_rumors_acc'])
        cog = 1.0 / acc 
        
        # Mat: 暂时随机生成，模拟对特定话题的兴趣
        mat = np.random.random()
        
        params[str(node_id)] = {
            "Per": per, "Pos": pos, "Mat": mat, "Cog": cog
        }
    return params

def add_latent_edges(G, agent_params, k=2):
    """
    基于 Per, Pos, Mat 使用 KD-Tree 添加潜在边 (Latent Edges)
    """
    nodes = sorted(list(G.nodes()))
    features = []
    for n in nodes:
        p = agent_params[str(n)]
        features.append([p['Per'], p['Pos'], p['Mat']])
    
    X = np.array(features)
    tree = KDTree(X)
    
    # 为每个节点寻找最近的 k 个“潜在邻居”
    dist, ind = tree.query(X, k=k+1) # +1 因为包含自身
    
    added_edges = 0
    for i, neighbors in enumerate(ind):
        u = nodes[i]
        for idx in neighbors[1:]: # 跳过自身
            v = nodes[idx]
            if not G.has_edge(u, v):
                G.add_edge(u, v, type='latent') # 标记为潜在边
                added_edges += 1
    
    print(f"Added {added_edges} latent edges to the network.")
    return G

def prepare_environment(source_agents_file, output_dir):
    # 1. 读取原始 Agent 数据
    with open(source_agents_file, 'r') as f:
        agents = json.load(f)
        
    # 2. 创建基础图 (假设是全连接或读取现有图，这里简化为从 agents_100.json 构建空图逻辑)
    # 实际应结合 rumor_test_env.py 中的建图逻辑
    # 这里仅做演示，假设图结构已知或由 env_create_* 生成
    # 我们先生成参数
    
    # 模拟一个图用于计算参数
    G = nx.Graph()
    G.add_nodes_from([int(k) for k in agents.keys()])
    # 随机加边模拟原始连接 (实际应读取 .graphml)
    for i in range(len(agents)):
        G.add_edge(i, (i+1)%len(agents))
    
    # 3. 计算参数
    params = calculate_agent_params(agents, G)
    
    # 4. 添加潜在边
    G_enhanced = add_latent_edges(G, params)
    
    # 5. 保存增强后的 Agent 配置 (合并参数)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for uid, agent in agents.items():
        agent.update(params[uid])
        # 保存到单独文件，兼容 Hu 的读取方式
        agent_path = os.path.join(output_dir, f'agent_{uid}')
        os.makedirs(agent_path, exist_ok=True)
        with open(os.path.join(agent_path, f'agent_{uid}.json'), 'w') as f:
            json.dump(agent, f, indent=4, cls=NumpyEncoder)
            
    # 保存图结构
    nx.write_graphml(G_enhanced, os.path.join(output_dir, 'social_graph_enhanced.graphml'))
    print(f"Environment prepared in {output_dir}")

# 使用示例 (需在主脚本中调用)
# prepare_environment('agents_100.json', 'Env_Hybrid_Test')