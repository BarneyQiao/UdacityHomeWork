from frozenlake import FrozenLakeEnv
import numpy as np
from plot_utils import plot_values
import copy

env = FrozenLakeEnv()
# # ========================1.迭代策略评估 ==========================
# V_list = []
# counts = 2
#
# """
# 用户迭代策略评估：
# 输入：环境、策略、gamma、theta
# 输出：V（状态价值函数）
# """
# def policy_evaluation(env,policy,gamma=1,theta=1e-8):
#     V = np.zeros(env.nS)
#     print(V)
#     V_list.append(V.copy())
#     while True:
#     # for i in range(counts):
#         delta = 0
#         for s in range(env.nS):
#             Vs = 0
#             for a,action_prob in enumerate(policy[s]):
#                 for prob,next_state,reward,done in env.P[s][a]:
#                     Vs += action_prob * prob * (reward + gamma * V[next_state])
#             delta = max(delta,np.abs(Vs - V[s]))  #这里其实就是在找 每个状态更新幅度最大的那个值，最大都没超过theta就说明收敛了
#             V[s] = Vs
#         V_list.append(V.copy())  # 测试
#         if delta < theta:
#             break
#     print(V)
#     return V
#

# random_policy = np.ones([env.nS,env.nA])/env.nA
# V = policy_evaluation(env,random_policy)
#
# # plot_values(V)
# print(len(V_list))
#
# for i in range(counts+1):
#     plot_values(V_list[i])
#
#
# # ========================2.v派->q派 动作值函数的评估==========================
# """
# 输入：环境、状态值函数估值、某一个状态、gamma
# 返回：q(s,a)
#
# 所谓Q函数，其实就是(s,a)对应的价值
# """
def q_from_v(env, V, s, gamma=1):
    q = np.zeros(env.nA)  # 返回的q应该是个一位数组，长度为动作空间维度，初始化为0
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[s][a]:
            q[a] += prob * (reward + gamma * V[next_state])

    return q
#
# Q = np.zeros([env.nS,env.nA])
#
# for s in range(env.nS):
#     Q[s] =q_from_v(env,V,s)
# print('Action Values:')
# print(Q)
#
#
# # ===========3.策略改进 ==============
# """
# 策略改进：
# 输入：环境，对该策略V估值，gamma
# 输出：policy（s行 a列）
# """
def policy_improvement(env, V, gamma=1):
    policy = np.zeros([env.nS,env.nA])/env.nA #设置一个全0的policy作为初始化

    for s in range(env.nS):
        q = q_from_v(env, V, s, gamma)

        #1 贪婪算法，直接选最大的q值 确定性策略
        # policy[s][np.argmax(q)] = 1
        #2 随机算法，对多个取得最大值的q值取一个随机
        best_a = np.argwhere(q==np.max(q)).flatten()
        policy[s] = np.sum([np.eye(env.nA)[i] for i in best_a], axis=0)/len(best_a)

    return policy
#
# policy = policy_improvement(env, V)
# print("Policy Improvement:")
# print(policy)
#
# #LEFT = 0 DOWN = 1 RIGHT = 2 UP = 3
#
# # =========== 4. 策略迭代 ===========
# """
# 输入：env环境、gamma、theta
# 输出：policy、V
# """
# def policy_iteration(env, gamma=1, theta = 1e-8):
#     policy = np.ones([env.nS,env.nA]) #一般初始化策略就是各个动作概率都一样
#     while True:
#         V = policy_evaluation(env,policy,gamma,theta) # 先进行迭代策略评估，得到当前策略的V
#         new_policy = policy_improvement(env,V)  #估计好了当前策略的V，就可以进行策略改进了，这里产生新策略
#
#         if(new_policy == policy).all():  #循环这个两个过程，直倒更新不懂了
#             break
#
#         # if np.max(abs(policy_evaluation(env, policy) - policy_evaluation(env, new_policy))) < theta*1e2:
#         #    break;
#
#         policy = copy.copy(new_policy)  # 拷贝新polcy，继续下一次循环
#
#     return policy,V   # 返回训练好的policy 和对应的 V估值
#
# policy_pi, V_pi = policy_iteration(env)
# print('\nOptimal Policy:LEFT = 0 DOWN = 1 RIGHT = 2 UP = 3')
# print(policy_pi)
#
# plot_values(V_pi)


# -=================5. 截断策略迭代 ==============
# def truncated_policy_evaluation(env, policy, V, max_it=1, gamma=1):
#     num_it = 0
#     while num_it < max_it:
#         for s in range(env.nS):
#             v = 0
#             q = q_from_v(env, V, s, gamma)
#             for a, action_prob in enumerate(policy[s]):
#                 v += action_prob * q[a]    #与值迭代关键不同的地方
#             V[s] = v
#             # 区别就在这 原来迭代策略的循环停止条件是更新值小，这里给定循环次数了
#         num_it += 1
#     return V
#
#
# def truncated_policy_iteration(env,max_it = 1,gamma = 1,theta = 1e-8):
#     V = np.zeros(env.nS)
#     policy = np.zeros([env.nS, env.nA]) / env.nA
#     while True:
#         policy = policy_improvement(env, V)
#         old_V = copy.copy(V)
#         V = truncated_policy_evaluation(env, policy, V, max_it, gamma)
#         if max(abs(V - old_V)) < theta:
#             break
#     return policy, V
#
# policy_tpi, V_tpi = truncated_policy_iteration(env, max_it=2)

# print the optimal policy
# print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
# print(policy_tpi,"\n")

# plot the optimal state-value function
# plot_values(V_tpi)

# ==========================6. 值迭代 ===============================
"""
值迭代：
输入：env,gamma,theta
输出：policy,V
其实输入输出和策略迭代和截断策略迭代是一样的
"""
def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            V[s] = max(q_from_v(env, V, s, gamma)) # 与策略迭代的关键不同
            delta = max(delta, abs(V[s]-v))
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma)
    return policy, V

policy_vi, V_vi = value_iteration(env)
# print the optimal policy
print("\nOptimal Policy (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3):")
print(policy_vi,"\n")

# plot the optimal state-value function
plot_values(V_vi)
