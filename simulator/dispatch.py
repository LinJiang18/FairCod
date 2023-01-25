
class DispatchSolution:

    def __init__(self):
        self.state = None
        self.action = None
        self.reward = None
        self.nextState = None

    def add_state(self,state):
        self.state = state  # np.array类型（骑手数目 * 匹配特征）

    def add_action(self,action):
        self.action = action  # 标量

    def add_reward(self,reward):
        self.reward = reward # 奖励

    def add_nextState(self,nextState):
        self.nextState = nextState
