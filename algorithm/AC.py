import torch
import numpy as np
import torch.nn.functional as F
import collections
import random


device = torch.device("cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, stateDim,actionDim):  # 200 + 3 + 20, 6
        super(PolicyNet, self).__init__()
        self.stateDim = stateDim   # 骑手状态
        self.actionDim = actionDim  # 动作状态
        self.S = torch.nn.Linear(stateDim, 64).to(device)
        self.A = torch.nn.Linear(actionDim,4).to(device)
        self.L1 = torch.nn.Linear(64+4,32).to(device)
        self.L2 = torch.nn.Linear(32,8).to(device)
        self.f = torch.nn.Linear(8, 1).to(device)

    def forward(self, X):
        s = X[:, :self.stateDim]
        a = X[:, -self.actionDim:]
        s1 = F.relu(self.S(s))
        a1 = F.relu(self.A(a))
        y1 = torch.cat((s1, a1), dim=1)
        l1 = F.relu((self.L1(y1)))
        l2 = F.relu((self.L2(l1)))
        return self.f(l2)


# critic
class ValueNet(torch.nn.Module):
    def __init__(self, stateDim,actionDim): # 200 + 3 +20
        super(ValueNet, self).__init__()
        self.stateDim = stateDim  # 骑手状态
        self.actionDim = actionDim  # 动作状态
        self.S = torch.nn.Linear(stateDim, 64).to(device)
        self.A = torch.nn.Linear(actionDim, 4).to(device)
        self.L1 = torch.nn.Linear(64 + 4, 32).to(device)
        self.L2 = torch.nn.Linear(32, 8).to(device)
        self.f = torch.nn.Linear(8, 1).to(device)

    def forward(self, X):
        s = X[:, :self.stateDim]
        a = X[:, -self.actionDim:]
        s1 = F.relu(self.S(s))
        a1 = F.relu(self.A(a))
        y1 = torch.cat((s1, a1), dim=1)
        l1 = F.relu((self.L1(y1)))
        l2 = F.relu((self.L2(l1)))
        return F.relu(self.f(l2))

class ActorCritic:
    def __init__(self, stateDim, actionDim, actorLr, criticLr, gamma, batchSize,device):
        self.actor = PolicyNet(stateDim,actionDim).to(device)
        self.critic = ValueNet(stateDim,actionDim).to(device)
        self.actorLr = actorLr
        self.criticLr = criticLr
        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actorLr)
        self.criticOptimizer = torch.optim.Adam(self.critic.parameters(), lr=self.criticLr)
        self.gamma = gamma
        self.stateDim = stateDim
        self.batchSize = batchSize

    # actor:采取动作
    def take_action(self, state):  # 训练
        state = torch.tensor(state, dtype=torch.float).to(device)
        vOutput = self.actor(state)
        vOutput = vOutput.reshape(-1)  # 将二维张量变为一维张量
        actionProb = torch.softmax(vOutput, dim=0)
        a = random.random()
        if a < 1:
            actionDist = torch.distributions.Categorical(actionProb)
            action = actionDist.sample().cpu()
        else:
            action = torch.max(actionProb,0)[1]
        return action.item()  # 对softmax函数求导时的用法


    def take_action_random(self, state):  # 训练
        state = torch.tensor(state, dtype=torch.float).to(device)
        a = random.sample(list(range(state.shape[0])),1)[0]
        # c_id_ = action_dist.sample()
        # print(c_id_)
        return a # 对softmax函数求导时的用法

    '''
    def take_next_action(self, _state):
        state_ = torch.tensor(_state, dtype=torch.float).to(device)
        v_output = self.actor(state_)
        v_output = v_output.reshape(-1)  # 将二维张量变为一维张量
        action_prob = torch.softmax(v_output, dim=0)
        c_id_ = torch.argmax(action_prob)
        # c_id_ = action_dist.sample()
        # print(c_id_)
        return c_id_.item()  # 对softmax函数求导时的用法

    def take_action_epsilon(self, _state, epsilon):
        state_ = torch.tensor(_state, dtype=torch.float).to(device)
        v_output = self.actor(state_)
        softmax_V = torch.softmax(v_output, dim=0)
        c_id = np.argmax(np.array(v_output.cpu().detach().numpy()))
        # epsilon-greedy 策略
        action_prob = np.ones(len(_state))
        action_prob = action_prob * (1 - epsilon) / (len(_state))
        action_prob[c_id] += epsilon
        c_id_ = np.argmax(np.random.multinomial(1, action_prob))
        return c_id_.item(), softmax_V.detach()
    '''

    def update(self, state,action,reward,nextState,round):
        #state = [torch.tensor(s, dtype=torch.float).to(device) for s in state]   # state目前是一个
        action = torch.tensor(action).view(-1,1).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        nextState = torch.tensor(nextState, dtype=torch.float).to(device)


        tdTarget = torch.unsqueeze(reward, 1) + self.gamma * self.critic(nextState).to(device)
        stateOne = [state[i][action[i]] for i in range(self.batchSize)]
        stateOne = torch.tensor(stateOne,dtype=torch.float).to(device)
#       stateOne = stateOne[:,:self.stateDim]
        V = self.critic(stateOne).to(device)
        criticLoss = torch.mean(F.mse_loss(V, tdTarget.detach())).to(device)  # TD error
        tdDelta = tdTarget - V  # 时序差分误差 TD_error

        logProb = torch.tensor([])
        for i in range(self.batchSize):
            stateTwo = torch.tensor(state[i],dtype=torch.float).to(device)
            lP = torch.log(torch.softmax(self.actor(stateTwo), dim=0)[action[i]]).to(device)
            logProb = torch.cat((logProb,lP),0).to(device)
        #stateTwo = torch.tensor(state, dtype=torch.float).to(device)
        #logProb = torch.log(torch.softmax(self.actor(stateTwo).squeeze(), dim=1).gather(1, action)).to(device)
        #log_probs = torch.log(self.actor(state).gather(1, action))
        actorLoss = torch.mean(-logProb * tdDelta.detach()).to(device)

        if round % 5 == 0:
            self.reset_learning_rate()

        self.criticOptimizer.zero_grad()
        self.actorOptimizer.zero_grad()
        criticLoss.backward()  # 计算critic网络的梯度
#       print(f'Critic Loss: {criticLoss}')
        actorLoss.backward()  # 计算actor网络的梯度
#       print(f'Actor Loss: {actorLoss}')
        self.criticOptimizer.step()  # 更新critic网络参数
        self.actorOptimizer.step()  # 更新actor网络参数

    def reset_learning_rate(self):
        self.criticLr = self.criticLr / 5
        self.criticOptimizer.param_groups[0]['lr'] = self.criticLr



class ReplayBuffer:
    def __init__(self, capacity,batchSize):
        self.buffer = collections.deque(maxlen=capacity)
        self.batchSize = batchSize

    def add(self, state,action,reward, nextState):
        self.buffer.append((state,action, reward, nextState))

    def sample(self):
        transitions = random.sample(self.buffer, self.batchSize)
        state, action, reward, nextState = zip(*transitions)
        state = list(state)
        state = [x.tolist() for x in state]
        return state, action, reward, np.array(nextState)

    def size(self):
        return len(self.buffer)