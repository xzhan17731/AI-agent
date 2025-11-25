"""
环境构建与测试工具集 (rumor_test_env.py)

此模块包含用于生成多种社交网络拓扑、创建 agent 配置、以及构造
谣言和帖子样本的辅助函数。目标是为多智能体谣言传播实验提供可复现
的环境（graphml、agent json、rumor/post 列表等）。

常见用法：
- 调用 `create_env1/2/3/create_env_fb` 生成保存到磁盘的实验环境
- 使用 `env_create_agent_test_*` 系列函数构造不同拓扑下的 agent 数据

注意：许多函数会创建或覆盖 `Saving_path` 下的 `agent_{i}` 目录。
"""

from prompt_env1 import *
from LLM import *
from sre_constants import error
import random
import os
import json
import re
import copy
import numpy as np
import shutil
import time
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# 默认的 agent 数量常量，可被 env_create_* 调用覆盖
NUM_OF_AGENTS = 100

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#[TODO] replace the hard-coded definition with external files
def create_rumors_test():
  """
  返回一个示例谣言字典，用于测试或初始化实验环境。

  返回格式为 {id: 文本}。函数目前包含若干示例谣言，可根据实验需要替换或
  从外部文件加载（TODO）。
  """

  rumor_list = {
      0: 'Nicolae Ceausescu does not died!',
      3: 'A living dinosaur is found in Yellowstone National Park.',
      2: 'Large Language Models are manned by real people acting as agents.',
      1: 'Drinking 3 ale a day can heal the cancer!',
  }

  return rumor_list

def read_facebook_network(id):
  """
  从仓库的 `facebook` 子目录读取指定 id 的社交网络（.edges、.feat、.egofeat），
  构建 NetworkX 图并返回。

  说明：
  - `id` 对应样本文件名，例如 `facebook/686.edges`。
  - 函数会打印节点/边数量，并绘制&保存网络图像到当前目录。
  - 返回值为 NetworkX 的 Graph 对象，节点未必有 'label' 属性（如果有会被用于绘图）。
  """

  dir = 'facebook'
  edges_path = f'{dir}/{id}.edges'
  with open(edges_path, 'r') as file:
      edges = [tuple(map(int, line.strip().split())) for line in file]

  # 使用 NetworkX 创建无向图并加入边
  G = nx.Graph()
  G.add_edges_from(edges)

  # 打印基本信息以便调试
  print("Number of nodes:", G.number_of_nodes())
  print("Number of edges:", G.number_of_edges())

  # 读取节点特征（若存在）并展示前几行
  feat_path = f'{dir}/{id}.feat'
  features = pd.read_csv(feat_path, sep=' ', header=None)
  features.columns = ['node'] + [f'feat_{i}' for i in range(1, features.shape[1])]
  print(features.head())

  # 读取自我特征（ego features），通常是 ego 节点的特征向量
  egofeat_path = f'{dir}/{id}.egofeat'
  egofeatures = pd.read_csv(egofeat_path, sep=' ', header=None)
  print(egofeatures.head())

  # 可视化并保存图像（便于人工检查拓扑）
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=6, edge_color='k', font_weight='bold')
  plt.title('Facebook Social Network')
  plt.savefig("Facebook_Social_Graph.png")
  plt.show()

  return G

def create_random_posts_test(num = 10):
  """
  生成一个示例的随机帖子字典，默认包含 50 条日常生活类的短文案。

  参数:
  - num: 返回的帖子数（上限为内置样本数）

  返回格式: {id: 文本}
  """

  posts_list = {
      0: 'Had the best cup of coffee this morning; it made my day!',
      1: 'Spent the afternoon reading a good book by the park.',
      2: 'The sound of rain is so calming. Perfect weather to stay in and relax.',
      3: 'Today is a lovely day for a long walk around the neighborhood.',
      4: 'Tried a new recipe today, and it actually turned out amazing!',
      5: 'Nothing beats the smell of fresh bread from a local bakery.',
      6: 'Ran into an old friend today—totally made my week!',
      7: 'Started journaling again; it feels good to put thoughts on paper.',
      8: 'Took my dog for a walk by the lake; he was so happy to explore.',
      9: 'Found a cozy little café around the corner; it might be my new favorite spot.',
      10: 'Ended the day with a gorgeous sunset. Feeling grateful.',
      11: 'Finally organized my closet; it feels like a fresh start!',
      12: 'Caught a beautiful sunrise this morning. Worth getting up early for!',
      13: 'Met a stranger who gave me great advice without even realizing it.',
      14: 'Tried painting for the first time—turns out it’s really relaxing!',
      15: 'Went for a bike ride around town; felt like a mini adventure.',
      16: 'Cooked dinner with friends; nothing beats a good meal and laughter.',
      17: 'Spent the afternoon at a museum. So inspiring to see all that art.',
      18: 'Found an old photo album today—brought back so many memories!',
      19: 'Did a random act of kindness today; feels good to brighten someone’s day.',
      20: 'Took a break from screens and went for a nature walk. Much needed!',
      21: 'Watched an old movie that reminded me of my childhood. Nostalgia overload!',
      22: 'Planted some flowers in the garden; can’t wait to see them bloom.',
      23: 'Learned a new word today and used it in a conversation. Feels rewarding!',
      24: 'Cleaned up my workspace, and now I feel so much more productive.',
      25: 'Spent time stargazing tonight; the sky was absolutely breathtaking.',
      26: 'Had a great conversation with a family member I don’t see often.',
      27: 'Visited a local farmer’s market and bought the freshest produce.',
      28: 'Listened to my favorite album from start to finish. What a mood lifter!',
      29: 'Tried yoga for the first time—my body feels so stretched and relaxed.',
      30: 'Found a handwritten letter from years ago; what a special moment.',
      31: 'Had a spontaneous dance party in the living room. Pure joy!',
      32: 'Spent the day volunteering; it’s incredible how rewarding helping others can be.',
      33: 'Picked up a new hobby today—let’s see how long this one lasts!',
      34: 'Made a playlist of songs from my favorite decade. Instant good vibes!',
      35: 'Went to the library and discovered a hidden gem of a book.',
      36: 'Caught a rainbow after the rain; what a magical moment.',
      37: 'Had an interesting chat with a stranger about life and dreams.',
      38: 'Tried a unique dessert today—surprisingly delicious!',
      39: 'Went for a drive with no destination in mind; sometimes it’s about the journey.',
      40: 'Wrote a thank-you note to someone who’s been kind to me.',
      41: 'Sat by the fireplace with a cup of hot chocolate. Cozy vibes all around.',
      42: 'Visited a nearby town I’ve never explored before; it felt like a mini vacation.',
      43: 'Caught up on a podcast I’ve been meaning to listen to. Learned so much!',
      44: 'Helped a neighbor carry their groceries. A small act but it felt good.',
      45: 'Went to a comedy show and laughed until my cheeks hurt.',
      46: 'Rearranged my living room—it feels like a whole new space now.',
      47: 'Took a nap in the afternoon and woke up feeling refreshed.',
      48: 'Spent the evening sketching; I’m not an artist, but it was fun!',
      49: 'Had a heartfelt phone call with an old friend. So much love and gratitude.',
  }


  # 根据请求数量截取样本并返回
  return {i: posts_list[i] for i in range(min(num, len(posts_list)))}

def env_create_agent_test_facebook(Saving_path, id = 686):
  """
  从 Facebook 格式数据构建实验 agent 文件：
  - 读取 `agents_170.json` 作为 agent 模板
  - 读取指定 id 的 .edges 文件构建图并重标号（0..N-1）
  - 将图中每个节点的朋友列表写入对应 agent 的 JSON，并保存到 `Saving_path/agent_{i}`

  此函数会为度数最高的前 5 个节点设置较高的谣言接受/传播能力（作为示例）。
  """

  with open('agents_170.json', 'r') as file:
    agent_list = json.load(file)

  dir = 'facebook'
  edges_path = f'{dir}/{id}.edges'
  with open(edges_path, 'r') as file:
      edges = [tuple(map(int, line.strip().split())) for line in file]

  # 使用 NetworkX 创建图并添加边
  G = nx.Graph()
  G.add_edges_from(edges)

  # 打印信息并重标号节点（确保节点从 0 开始连续）
  print("Number of nodes:", G.number_of_nodes())
  print("Number of edges:", G.number_of_edges())

  mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes)}
  G = nx.relabel_nodes(G, mapping)

  # 使用 agent_list 的名称作为节点标签，便于繪图显示
  count = 0
  for node in G.nodes:
      G.nodes[node]['label'] = agent_list[str(count)]['agent_name']
      count += 1

  # 可视化并保存
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=500, edge_color='k', font_weight='bold')
  plt.title('Social Network')
  plt.savefig(f"Social_Graph_FB_{id}.png")
  plt.show()

  # 将图写出为 graphml，便于在外部工具加载
  file_path = f"Social_Graph_FB_{id}.graphml"
  nx.write_graphml(G, file_path)

  # 计算度数并找到前 5 位高连接节点
  degrees = {i: len(list(G.neighbors(i))) for i in range(G.number_of_nodes())}
  top_5_agents = sorted(degrees, key=degrees.get, reverse=True)[:5]

  # 将朋友列表写入各 agent 配置并保存到磁盘
  for i in range(G.number_of_nodes()):
    agent = agent_list[str(i)]
    friends = list(G.neighbors(i))
    agent['friends'] = friends

    # 示例性地给高连接节点设置更强的谣言接收/传播能力
    if i in top_5_agents:
        agent['agent_rumors_acc'] = '4'
        agent['agent_rumors_spread'] = '3'

    # 每次都确保目标目录存在（先删除再创建，保证干净）
    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4, cls=NumpyEncoder)


def env_create_agent_test_sc(num, Saving_path):
  """
  生成一个带有优先连接（Scale-Free / preferential attachment）特性的社交图并将
  agent 信息写出到 `Saving_path`。

  实现方式：
  - 读取 `agents_100.json` 作为 agent 模板
  - 前 4 个节点构造一个完全连接子图以作为种子
  - 之后的节点按 preferential attachment 规则连接到已有节点（通过度数比例）
  - 将图保存为 image 与 graphml，并把每个 agent 的朋友列表写入 JSON
  """

  with open('agents_100.json', 'r') as file:
    agent_list = json.load(file)

  # Relation Graph
  G = nx.Graph()
  for i in range(num):
    G.add_node(i, label=agent_list[str(i)]['agent_name'])

  # seed: 前 4 个节点构成完全图
  for i in range(4):
      for j in range(i+1, 4):
          G.add_edge(i, j)

  # 根据 preferential attachment 添加边
  j = 4
  np.random.seed(66)
  while j < num:
      k = j

      total_degree = sum(dict(G.degree()).values())
      nodes = [node for node in G.nodes() if node != k]
      probs = [G.degree(node) / total_degree for node in nodes]

      ns = np.random.choice(nodes, size=4, replace=False, p=probs)
      for n in ns:
          if n != k and not G.has_edge(n, k):
              G.add_edge(n, k)
      j += 1

  # 绘图并保存
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=500, edge_color='k', font_weight='bold')
  plt.title('Social Network')
  plt.savefig("Social_Graph_sc.png")
  plt.show()

  file_path = "Social_Graph_sc.graphml"
  nx.write_graphml(G, file_path)

  # 将朋友列表写到 agent 配置并保存
  degrees = {i: len(list(G.neighbors(i))) for i in range(num)}
  top_5_agents = sorted(degrees, key=degrees.get, reverse=True)[:5]
  for i in range(num):
    agent = agent_list[str(i)]
    friends = list(G.neighbors(i))
    agent['friends'] = friends

    # 给高连接节点示例性地设置更强能力
    if i in top_5_agents:
        agent['agent_rumors_acc'] = '4'
        agent['agent_rumors_spread'] = '3'

    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4, cls=NumpyEncoder)

def env_create_agent_test_random(num, Saving_path):
  """
  使用随机策略构造一个社交图：
  - 以 `agents_100.json` 为模板创建节点
  - 先构造一个小型完全子图作为种子，然后以随机采样的方式添加边直到达到目标边数
  - 将结果保存为图像和 graphml，并写出每个 agent 的朋友列表
  """

  with open('agents_100.json', 'r') as file:
    agent_list = json.load(file)

  # Relation Graph
  G = nx.Graph()
  for i in range(num):
    G.add_node(i, label=agent_list[str(i)]['agent_name'])

  # seed 完全子图
  for i in range(4):
      for j in range(i+1, 4):
          G.add_edge(i, j)

  # 随机添加边直到达到指定数量（这里以 edge_num < 390 为例）
  j = 4
  np.random.seed(66)
  edge_num = 0
  p = 0.6
  while edge_num < 390:
    u, v = random.sample(list(G.nodes()), 2)

    if not G.has_edge(u, v):
      if random.random() < p:
        G.add_edge(u, v)
        edge_num += 1

  # 绘图并保存
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=500, edge_color='k', font_weight='bold')
  plt.title('Social Network')
  plt.savefig("Social_Graph_random.png")
  plt.show()

  file_path = "Social_Graph_random.graphml"
  nx.write_graphml(G, file_path)

  # 将朋友列表写入每个 agent 的 json 并保存
  for i in range(num):
    agent = agent_list[str(i)]
    friends = list(G.neighbors(i))
    agent['friends'] = friends

    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4, cls=NumpyEncoder)

def env_create_agent_test_small_world(num, Saving_path):
  """
  使用 Watts-Strogatz 小世界模型生成社交网络，并写出 agent 数据到磁盘。

  参数：
  - num: 节点数量
  - Saving_path: 输出目录
  """

  with open('agents_100.json', 'r') as file:
    agent_list = json.load(file)

  np.random.seed(66)

  # Generate a small-world network using the Watts-Strogatz model.
  nearest_neighbors = 4  # Each node is connected to 4 nearest neighbors
  rewiring_prob = 0.3
  G = nx.watts_strogatz_graph(num, nearest_neighbors, rewiring_prob)

  for i in range(num):
    G.nodes[i]['label'] = agent_list[str(i)]['agent_name']

  # 绘图并保存
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=500, edge_color='k', font_weight='bold')
  plt.title('Social Network')
  plt.savefig("Social_Graph_sw.png")
  plt.show()

  # Save the graph
  file_path = "Social_Graph_sw.graphml"
  nx.write_graphml(G, file_path)

  # Update agents with their friends list and save their data
  for i in range(num):
    agent = agent_list[str(i)]
    friends = list(G.neighbors(i))
    agent['friends'] = friends

    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4, cls=NumpyEncoder)

def env_create_agent_test(num, Saving_path):
  """
  为小规模测试创建一个手工定义的 agent 列表（示例数据）。n
  该函数用于调试或演示场景，其中 agent 属性是预先设定的。
  """

  agent_list = {
      0: {
          'agent_name': 'Keqing',
          'agent_age': '18',
          'agent_job': 'Policeman',
          'agent_traits': 'Ambitious',
          'agent_rumors_acc': '3',
          'agent_rumors_spread': '3',
      },
      1: {
          'agent_name': 'Radu',
          'agent_age': '48',
          'agent_job': 'Teacher',
          'agent_traits': 'Calm, Brave',
          'agent_rumors_acc': '2',
          'agent_rumors_spread': '2',
      },
      2: {
          'agent_name': 'Karen',
          'agent_age': '22',
          'agent_job': 'Waiter',
          'agent_traits': 'Gregarious',
          'agent_rumors_acc': '4',
          'agent_rumors_spread': '3',
      },
      3: {
          'agent_name': 'Leo',
          'agent_age': '35',
          'agent_job': 'Software Developer',
          'agent_traits': 'Analytical, Persistent',
          'agent_rumors_acc': '3',
          'agent_rumors_spread': '3',
      },
      4: {
          'agent_name': 'Hana',
          'agent_age': '27',
          'agent_job': 'Graphic Designer',
          'agent_traits': 'Creative, Detail-oriented',
          'agent_rumors_acc': '2',
          'agent_rumors_spread': '3',
      },
      5: {
          'agent_name': 'Ismail',
          'agent_age': '44',
          'agent_job': 'Doctor',
          'agent_traits': 'Empathetic, Resilient',
          'agent_rumors_acc': '3',
          'agent_rumors_spread': '3',
      },
      6: {
          'agent_name': 'Elena',
          'agent_age': '31',
          'agent_job': 'Journalist',
          'agent_traits': 'Inquisitive, Bold',
          'agent_rumors_acc': '4',
          'agent_rumors_spread': '3',
      },
      7: {
          'agent_name': 'Omar',
          'agent_age': '40',
          'agent_job': 'Chef',
          'agent_traits': 'Innovative, Patient',
          'agent_rumors_acc': '4',
          'agent_rumors_spread': '3',
      },
      8: {
          'agent_name': 'Jessica',
          'agent_age': '24',
          'agent_job': 'Flight Attendant',
          'agent_traits': 'Sociable, Energetic',
          'agent_rumors_acc': '3',
          'agent_rumors_spread': '3',
      },
      9: {
          'agent_name': 'Sam',
          'agent_age': '50',
          'agent_job': 'Engineer',
          'agent_traits': 'Practical, Methodical',
          'agent_rumors_acc': '2',
          'agent_rumors_spread': '2',
      },
      
  }

  # Relation Graph
  G = nx.Graph()
  for i in range(num):
    G.add_node(i, label=agent_list[i]['agent_name'])

  # 使用随机策略添加若干边，构建一个小型社交网络用于演示
  random.seed(4242)
  edge_num = 0
  p = 0.6
  while edge_num < 35:
    u, v = random.sample(list(G.nodes()), 2)

    if not G.has_edge(u, v):
      if random.random() < p:
        G.add_edge(u, v)
        edge_num += 1

  # 绘图并保存
  plt.figure(figsize=(8, 6))
  pos = nx.spring_layout(G, seed=42)
  nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue', node_size=500, edge_color='k', font_weight='bold')
  plt.title('Social Network')
  plt.savefig("Social_Graph.png")
  plt.show()

  # 保存 graphml
  file_path = "Social_Graph.graphml"
  nx.write_graphml(G, file_path)

  # 将朋友列表写入 agent 配置并保存
  for i in range(num):
    agent = agent_list[i]
    friends = list(G.neighbors(i))
    agent['friends'] = friends

    if not os.path.exists(Saving_path+f'/agent_{i}'):
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/agent_{i}')
      os.makedirs(Saving_path+f'/agent_{i}', exist_ok=True)

    with open(Saving_path+f'/agent_{i}/agent_{i}.json', 'w') as f:
        json.dump(agent, f, indent = 4, cls=NumpyEncoder)



def create_env1(Saving_path): # Random 10
  """
  创建实验环境 1（随机网络，默认 NUM_OF_AGENTS 节点）并把相关文件写入 `Saving_path`。
  """
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  # 生成 agent 配置
  env_create_agent_test_random(NUM_OF_AGENTS, Saving_path)

  # 创建并写入谣言与帖子列表
  rumor_list = create_rumors_test()
  with open(Saving_path+f'/rumor_list.json', 'w') as f:
    json.dump(rumor_list, f, indent = 4, cls=NumpyEncoder)

  posts_list = create_random_posts_test()
  with open(Saving_path+f'/posts_list.json', 'w') as f:
    json.dump(posts_list, f, indent = 4, cls=NumpyEncoder)

def create_env2(Saving_path): # Scale Free 20
  """
  创建实验环境 2（Scale-Free 网络示例），并写出需要的文件到 `Saving_path`。
  """
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  # 生成优先连接网络并写出 agent 数据
  env_create_agent_test_sc(100, Saving_path)

  # 写出谣言与帖子列表
  rumor_list = create_rumors_test()
  with open(Saving_path+f'/rumor_list.json', 'w') as f:
    json.dump(rumor_list, f, indent = 4, cls=NumpyEncoder)

  posts_list = create_random_posts_test(20)
  with open(Saving_path+f'/posts_list.json', 'w') as f:
    json.dump(posts_list, f, indent = 4, cls=NumpyEncoder)

def create_env3(Saving_path): # Scale Free 20
  """
  创建实验环境 3（小世界网络示例），并写出需要的文件到 `Saving_path`。
  """
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  # 生成小世界网络并写出 agent 数据
  env_create_agent_test_small_world(100, Saving_path)

  # 写出谣言与帖子列表
  rumor_list = create_rumors_test()
  with open(Saving_path+f'/rumor_list.json', 'w') as f:
    json.dump(rumor_list, f, indent = 4, cls=NumpyEncoder)

  posts_list = create_random_posts_test(20)
  with open(Saving_path+f'/posts_list.json', 'w') as f:
    json.dump(posts_list, f, indent = 4, cls=NumpyEncoder)

def create_env_fb(Saving_path, id = 686): # Scale Free 20
  """
  创建基于 Facebook 数据的实验环境（读取指定 id 的 Facebook 样例文件）。
  """
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  # 生成并写出 agent 数据（从 Facebook edges 与 agents_170.json）
  env_create_agent_test_facebook(Saving_path, id = id)

  # 写出谣言与帖子列表（示例取 50 条）
  rumor_list = create_rumors_test()
  with open(Saving_path+f'/rumor_list.json', 'w') as f:
    json.dump(rumor_list, f, indent = 4, cls=NumpyEncoder)

  posts_list = create_random_posts_test(50)
  with open(Saving_path+f'/posts_list.json', 'w') as f:
    json.dump(posts_list, f, indent = 4, cls=NumpyEncoder)
  

def main():
  """
  简单的入口函数示例：
  - 将 `Code_dir_path` 修改为你的项目路径后运行以创建环境文件（只需第一次运行）
  """
  Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/'  # Put the current code directory path here
  Saving_path = Code_dir_path + 'Env_Rumor_Test'

  # The first time to create the environment, after that you can comment it
  create_env2(Saving_path)
  #G = read_facebook_network(686)
  #create_env_fb(Saving_path, 686)

if __name__ == "__main__":
  main()