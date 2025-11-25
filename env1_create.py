# Box moving to target without collision

"""
env1_create.py

功能：提供一个简单的格子搬箱子（box-moving）环境构建工具和与之配套的
辅助函数（如状态生成、语法检查与动作执行等）。通常用于生成训练/测试
样例并在多智能体/强化学习实验中作为环境状态提供者。

注意：脚本末尾会调用 `create_env1` 生成若干场景文件，运行后会在指定路径
写入 `env_pg_state_*` 目录及其中的 JSON 文件；如果只想导入函数请在第一次运行
生成数据后注释掉底部的调用行以避免重复覆盖。
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

def surround_index_func(row_num, coloum_num, row_index, coloum_index):
  """
  计算指定位置周围的相邻位置索引
  
  参数:
  row_num: 行数
  coloum_num: 列数
  row_index: 当前行索引
  coloum_index: 当前列索引
  
  返回:
  surround_index_list: 包含上下左右相邻位置坐标的列表
  """
  surround_index_list = []
  # 遍历上下左右四个方向的位置
  for i, j in ([row_index-1, coloum_index], [row_index+1, coloum_index], [row_index, coloum_index-1], [row_index, coloum_index+1]):
    # 检查索引是否在合法范围内且不是当前位置本身
    if i>=0 and i<=row_num-1 and j>=0 and j<=coloum_num-1 and not (i == row_index and j == coloum_index):
      # 将合法的相邻位置加入列表（坐标加0.5是为了表示格子中心）
      surround_index_list.append([i+0.5,j+0.5])
  return surround_index_list

def state_update_func(pg_row_num, pg_column_num, pg_dict):
  """
  更新所有智能体的状态信息，生成状态更新提示
  
  参数:
  pg_row_num: 游戏区域行数
  pg_column_num: 游戏区域列数
  pg_dict: 当前游戏状态字典
  
  返回:
  state_update_prompt: 所有智能体的状态更新提示字符串
  """
  pg_dict_copy = copy.deepcopy(pg_dict)
  state_update_prompt = ''
  # 遍历每个格子
  for i in range(pg_row_num):
    for j in range(pg_column_num):
      # 获取当前格子中的物品列表
      square_item_list = pg_dict_copy[str(i+0.5)+'_'+str(j+0.5)]
      # 筛选出所有以'box'开头的物品（箱子）
      square_item_only_box = [item for item in square_item_list if item[:3]=='box']
      # 获取周围相邻位置列表
      surround_index_list = surround_index_func(pg_row_num, pg_column_num, i, j)
      # 添加当前智能体的观察信息
      state_update_prompt += f'Agent[{i+0.5}, {j+0.5}]: I am in square[{i+0.5}, {j+0.5}], I can observe {square_item_list}, I can do '
      action_list = []
      # 对于每个箱子，生成可能的动作列表
      for box in square_item_only_box:
        # 可以移动到相邻的四个位置
        for surround_index in surround_index_list:
          action_list.append(f'move({box}, square{surround_index})')
        # 如果同位置存在对应颜色的目标，则可以将箱子移入目标中
        if 'target'+box[3:] in square_item_list:
          action_list.append(f'move({box}, target{box[3:]})')
      state_update_prompt += f'{action_list}\n'
  return state_update_prompt

def state_update_func_local_agent(pg_row_num, pg_column_num, pg_row_i, pg_column_j, pg_dict):
  """
  分别更新本地智能体和其他智能体的状态信息
  
  参数:
  pg_row_num: 游戏区域行数
  pg_column_num: 游戏区域列数
  pg_row_i: 本地智能体行索引
  pg_column_j: 本地智能体列索引
  pg_dict: 当前游戏状态字典
  
  返回:
  state_update_prompt_local_agent: 本地智能体状态更新提示
  state_update_prompt_other_agent: 其他智能体状态更新提示
  """
  pg_dict_copy = copy.deepcopy(pg_dict)
  state_update_prompt_local_agent = ''
  state_update_prompt_other_agent = ''

  # 遍历所有格子，处理非本地智能体的信息
  for i in range(pg_row_num):
    for j in range(pg_column_num):
      # 排除本地智能体位置
      if not (i == pg_row_i and pg_column_j == j):
        square_item_list = pg_dict_copy[str(i+0.5)+'_'+str(j+0.5)]
        square_item_only_box = [item for item in square_item_list if item[:3]=='box']
        surround_index_list = surround_index_func(pg_row_num, pg_column_num, i, j)
        state_update_prompt_other_agent += f'Agent[{i+0.5}, {j+0.5}]: I am in square[{i+0.5}, {j+0.5}], I can observe {square_item_list}, I can do '
        action_list = []
        # 生成其他智能体可执行的动作列表
        for box in square_item_only_box:
          for surround_index in surround_index_list:
            action_list.append(f'move({box}, square{surround_index})')
          if 'target'+box[3:] in square_item_list:
            action_list.append(f'move({box}, target{box[3:]})')
        state_update_prompt_other_agent += f'{action_list}\n'

  # 处理本地智能体信息
  square_item_list = pg_dict_copy[str(pg_row_i+0.5)+'_'+str(pg_column_j+0.5)]
  square_item_only_box = [item for item in square_item_list if item[:3]=='box']
  surround_index_list = surround_index_func(pg_row_num, pg_column_num, pg_row_i, pg_column_j)
  state_update_prompt_local_agent += f'Agent[{pg_row_i+0.5}, {pg_column_j+0.5}]: in square[{pg_row_i+0.5}, {pg_column_j+0.5}], can observe {square_item_list}, can do '
  action_list = []
  # 生成本地智能体可执行的动作列表
  for box in square_item_only_box:
    for surround_index in surround_index_list:
      action_list.append(f'move({box}, square{surround_index})')
    if 'target'+box[3:] in square_item_list:
      action_list.append(f'move({box}, target{box[3:]})')
  state_update_prompt_local_agent += f'{action_list}\n'
  return state_update_prompt_local_agent, state_update_prompt_other_agent

def with_action_syntactic_check_func(pg_dict_input, response, user_prompt_list_input, response_total_list_input, model_name, dialogue_history_method, cen_decen_framework):
  """
  对LLM响应进行语法检查和验证，确保动作格式正确且可行
  
  参数:
  pg_dict_input: 当前游戏状态字典
  response: LLM的响应
  user_prompt_list_input: 用户提示列表
  response_total_list_input: 历史响应列表
  model_name: 使用的LLM模型名称
  dialogue_history_method: 对话历史方法
  cen_decen_framework: 中心化或去中心化框架
  
  返回:
  response: 经过验证的响应或错误信息
  token_num_count_list_add: 使用的token数量列表
  """
  user_prompt_list = copy.deepcopy(user_prompt_list_input)
  response_total_list = copy.deepcopy(response_total_list_input)
  iteration_num = 0
  token_num_count_list_add = []
  # 最多迭代6次进行语法检查
  while iteration_num < 6:
    response_total_list.append(response)
    try:
      # 解析 LLM 返回的 JSON 字符串为字典（期望格式为 {"x_y": "move(box_blue, square[1.5, 2.5])", ...} ）
      original_response_dict = json.loads(response)

      pg_dict_original = copy.deepcopy(pg_dict_input)
      transformed_dict = {}
      # 解析响应中的动作指令
      # 遍历每个响应项，解析出坐标、要移动的物体和目标位置
      for key, value in original_response_dict.items():
        # key 示例: '1.5_2.5' -> 提取为浮点坐标 (1.5, 2.5)
        coordinates = tuple(map(float, re.findall(r"\d+\.?\d*", key)))

        # 使用正则匹配 move(item, location) 形式的动作字符串
        match = re.match(r"move\((.*?),\s(.*?)\)", value)
        if match:
          item, location = match.groups()

          # 若 location 包含 'square'，则将其解析为坐标元组；否则保留为字符串（可能是 target_x）
          if "square" in location:
            location = tuple(map(float, re.findall(r"\d+\.?\d*", location)))

          # transformed_dict 存储形式: {(x,y): [item, location]}
          transformed_dict[coordinates] = [item, location]

      feedback = ''
      # 验证每个动作是否合法
      for key, value in transformed_dict.items():
        # print(f"Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
        # 检查箱子是否在当前位置且目标位置是相邻格子
        if value[0] in pg_dict_original[str(key[0]) + '_' + str(key[1])] and type(value[1]) == tuple and (
                (np.abs(key[0] - value[1][0]) == 0 and np.abs(key[1] - value[1][1]) == 1) or (
                np.abs(key[0] - value[1][0]) == 1 and np.abs(key[1] - value[1][1]) == 0)):
          pass
        # 检查是否是将箱子放入匹配颜色的目标中
        elif value[0] in pg_dict_original[str(key[0]) + '_' + str(key[1])] and type(value[1]) == str and value[1] in \
                pg_dict_original[str(key[0]) + '_' + str(key[1])] and value[0][:4] == 'box_' and value[1][
                                                                                                 :7] == 'target_' and \
                value[0][4:] == value[1][7:]:
          pass
        else:
          # 动作不合法时添加反馈信息
          # print(f"Error, Iteration Num: {iteration_num}, Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
          # 若既不是移动到相邻格子，又不是颜色匹配的放入操作，则认为非法
          feedback += f'Your assigned task for {key[0]}_{key[1]} is not in the doable action list; '
    except:
      # 响应不是有效的JSON格式时抛出异常
      raise error(f'The response in wrong json format: {response}')
      feedback = 'Your assigned plan is not in the correct json format as before. If your answer is empty dict, please check whether you miss to move box into the same colored target like move(box_blue, target_blue)'

    # 如果有错误反馈，则要求重新规划动作
    if feedback != '':
      feedback += 'Please replan for all the agents again with the same ouput format:'
      print('----------Syntactic Check----------')
      print(f'Response original: {response}')
      print(f'Feedback: {feedback}')
      user_prompt_list.append(feedback)
      messages = message_construct_func(user_prompt_list, response_total_list, dialogue_history_method) # message construction
      print(f'Length of messages {len(messages)}')
      response, token_num_count = GPT_response(messages, model_name)
      token_num_count_list_add.append(token_num_count)
      print(f'Response new: {response}\n')
      if response == 'Out of tokens':
        return response, token_num_count_list_add
      iteration_num += 1
    else:
      # 没有错误则返回验证通过的响应
      return response, token_num_count_list_add
  return 'Syntactic Error', token_num_count_list_add

def action_from_response(pg_dict_input, original_response_dict):
  """
  根据响应执行具体动作，更新游戏状态
  
  参数:
  pg_dict_input: 当前游戏状态字典
  original_response_dict: 解析后的动作指令字典
  
  返回:
  system_error_feedback: 系统错误反馈信息
  pg_dict_original: 更新后的游戏状态字典
  """
  system_error_feedback = ''
  pg_dict_original = copy.deepcopy(pg_dict_input)
  transformed_dict = {}
  # 解析响应中的动作指令
  # 将原始响应（字符串形式）解析并转换为内部统一表示 (coordinates -> [item, location])
  for key, value in original_response_dict.items():
    # key 示例: '1.5_2.5' -> (1.5, 2.5)
    coordinates = tuple(map(float, re.findall(r"\d+\.?\d*", key)))

    # 解析 move(item, location) 的动作字符串
    match = re.match(r"move\((.*?),\s(.*?)\)", value)
    if match:
      item, location = match.groups()
      # 若 location 包含 'square'，将其解析为坐标元组；否则保留为字符串（target_x）
      if "square" in location:
          location = tuple(map(float, re.findall(r"\d+\.?\d*", location)))
      transformed_dict[coordinates] = [item, location]

  # 执行每个动作
  for key, value in transformed_dict.items():
    #print(f"Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
    # 移动箱子到相邻格子
    if value[0] in pg_dict_original[str(key[0])+'_'+str(key[1])] and type(value[1]) == tuple and ((np.abs(key[0]-value[1][0])==0 and np.abs(key[1]-value[1][1])==1) or (np.abs(key[0]-value[1][0])==1 and np.abs(key[1]-value[1][1])==0)):
      pg_dict_original[str(key[0])+'_'+str(key[1])].remove(value[0])
      pg_dict_original[str(value[1][0])+'_'+str(value[1][1])].append(value[0])
    # 将箱子放入匹配颜色的目标中
    elif value[0] in pg_dict_original[str(key[0])+'_'+str(key[1])] and type(value[1]) == str and value[1] in pg_dict_original[str(key[0])+'_'+str(key[1])] and value[0][:4] == 'box_' and value[1][:7] == 'target_' and value[0][4:] == value[1][7:]:
      pg_dict_original[str(key[0])+'_'+str(key[1])].remove(value[0])
      pg_dict_original[str(key[0])+'_'+str(key[1])].remove(value[1])
    else:
      # 动作非法时记录错误信息
      #print(f"Error, Iteration Num: {iteration_num}, Key: {key}, Value1: {value[0]}, Value2: {value[1]}")
      # 非法动作：既不是移动到相邻格子，也不是颜色匹配的放入
      system_error_feedback += f'Your assigned task for {key[0]}_{key[1]} is not in the doable action list; '

  return system_error_feedback, pg_dict_original

def env_create(pg_row_num = 5, pg_column_num = 5, box_num_low_bound = 2, box_num_upper_bound = 2, color_list = ['blue', 'red', 'green', 'purple', 'orange']):
  """
  创建环境，随机分配箱子和目标位置
  
  参数:
  pg_row_num: 游戏区域行数，默认为5
  pg_column_num: 游戏区域列数，默认为5
  box_num_low_bound: 每种颜色箱子数量下限，默认为2
  box_num_upper_bound: 每种颜色箱子数量上限，默认为2
  color_list: 颜色列表，默认包含5种颜色
  
  返回:
  pg_dict: 初始化的游戏状态字典
  """
  # pg_dict records the items in each square over steps, here in the initial setting, we randomly assign items into each square
  # pg_dict记录每个格子中的物品，初始时随机分配物品到各个格子
  pg_dict = {}
  # 初始化空的游戏区域
  for i in range(pg_row_num):
    for j in range(pg_column_num):
      pg_dict[str(i+0.5)+'_'+str(j+0.5)] = []

  # 为每种颜色随机放置箱子和目标
  for color in color_list:
    # 随机确定该颜色箱子的数量
    box_num = random.randint(box_num_low_bound, box_num_upper_bound)
    for _ in range(box_num):
      # 随机选择箱子放置位置
      N_box = random.randint(0, pg_row_num*pg_column_num - 1)
      a_box = N_box // pg_column_num
      b_box = N_box % pg_column_num
      # 随机选择目标放置位置
      N_target = random.randint(0, pg_row_num*pg_column_num - 1)
      a_target = N_target // pg_column_num
      b_target = N_target % pg_column_num
      # 在相应位置放置箱子和目标
      pg_dict[str(a_box+0.5)+'_'+str(b_box+0.5)].append('box_' + color)
      pg_dict[str(a_target+0.5)+'_'+str(b_target+0.5)].append('target_' + color)
  return pg_dict

def create_env1(Saving_path, repeat_num = 10):
  """
  创建完整的Env1环境，包括不同尺寸的多个测试场景
  
  参数:
  Saving_path: 环境保存路径
  repeat_num: 每种环境配置重复次数，默认为10次
  """
  # 如果保存路径不存在则创建，否则先删除再创建
  if not os.path.exists(Saving_path):
    os.makedirs(Saving_path, exist_ok=True)
  else:
    shutil.rmtree(Saving_path)
    os.makedirs(Saving_path, exist_ok=True)

  # 为不同的网格尺寸创建环境
  for i ,j in [(2,2), (2,4), (4,4), (4,8)]:

    # 为特定尺寸创建保存目录
    if not os.path.exists(Saving_path+f'/env_pg_state_{i}_{j}'):
      os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}', exist_ok=True)
    else:
      shutil.rmtree(Saving_path+f'/env_pg_state_{i}_{j}')
      os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}', exist_ok=True)

    # 创建指定数量的重复环境实例
    for iteration_num in range(repeat_num):
      # 定义游戏区域的行列数以及箱子数量范围
      pg_row_num = i; pg_column_num = j; box_num_low_bound = 1; box_num_upper_bound = 3
      # 定义使用的颜色列表
      color_list = ['blue', 'red', 'green', 'purple', 'orange']
      # 创建环境状态
      pg_dict = env_create(pg_row_num, pg_column_num, box_num_low_bound, box_num_upper_bound, color_list)
      # 创建当前环境实例的保存目录
      os.makedirs(Saving_path+f'/env_pg_state_{i}_{j}/pg_state{iteration_num}', exist_ok=True)
      # 将环境状态保存为JSON文件
      with open(Saving_path+f'/env_pg_state_{i}_{j}/pg_state{iteration_num}/pg_state{iteration_num}.json', 'w') as f:
        json.dump(pg_dict, f)

Code_dir_path = 'path_to_multi-agent-framework/multi-agent-framework/' # Put the current code directory path here
Saving_path = Code_dir_path + 'Env1_BoxNet1'
# The first time to create the environment, after that you can comment it
# 第一次创建环境时取消注释，之后可以注释掉
create_env1(Saving_path, repeat_num = 10)