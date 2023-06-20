import numpy as np
import pandas as pd
import torch
import pickle
from simulator.envs import *
from route_planning.route import *
from algorithm.AC import *
from simulator.dispatch import *
from simulator.utility import cal_distance,cal_best_route,cal_route_dir,process_memory


dayIndex = 0  # 当前第几轮
maxDay = 25 # 最大循环多少轮
maxTime = 179 # 派单的轮数
# hexagon grid

matrixX = 10
matrixY = 10
memorySize = 100000
batchSize = 50


# 读取数据部分


  # a+不会被覆写

quit = 0

realOrder = pd.read_pickle('../data/OrderList.pickle')  # 30天订单信息
# 订单信息:  ID,day,time,merchantNode,userNode,distance,promiseTime,price

courierInit = pd.read_pickle('../data/courierInit.pickle')  # 骑手初始化信息
# 骑手信息: ID,initialNode

env = Region(courierInit, realOrder, matrixX, matrixY, maxDay, maxTime + 1)

env.set_node_info()  # 设置一系列节点信息
env.set_courier_info()  # 设置骑手的信息

actionDim = 7  # 动作维度
stateDim = 223  # 状态维度
T = 0  # 一天的时间计数

# ac:0.001
#cr:0.0005
actorLr = 0.001
criticLr = 0.00005

gamma = 0.9
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
agent = ActorCritic(stateDim, actionDim, actorLr, criticLr, gamma, batchSize, device)
replayBuffer = ReplayBuffer(memorySize, batchSize)
dayRecorder = open('../info/day_reward.txt', 'a+')
dayRecorder.truncate(0)

while dayIndex < maxDay:  # 利用30天的订单

    originalOrderTime = 0
    realOrderTime = 0

    env.get_day_info(dayIndex)  # 获得这一天的天信息
    env.reset_clean()  # 将这一天的内容重置
    dayReward = []
    stepRecorder = open(f'../info/step_info/day{dayIndex}.txt', 'a+')
    stepRecorder.truncate(0)
    print(f'Day {dayIndex}:')
    T = 0


    while T < maxTime:  # 这一天的时间
        dDict = {}
        stepRecorder.write("step_info" + str(T) + ':' + ' ')
        for order in env.dayOrder[env.cityTime]:  # 输入这个时段的订单
            originalOrderTime += order.orderDistance * 4 + 3
            courierList = env.action_collect(order)  # 适合的骑手
            # 计算骑手状态
            courierStateArray = env.courier_state_compute(courierList)  # n(骑手数目) * m(骑手特征)  # 23
            # 订单订单状态
            orderStateArray = env.order_state_compute(courierList, order)  # n * k(订单特征)  # 6
            # 计算系统整体的供需关系
            supplydemandStateArray = env.sd_state_compute()  # 1 * 200 (供需特征) # 200

            stateArray = np.hstack((courierStateArray,
                                    supplydemandStateArray.reshape(1, 200).repeat(courierStateArray.shape[0], 0)))
            stateMatchArray = np.hstack((stateArray, orderStateArray))

            if dayIndex < 0:
                rewardList = []
                for i in range(orderStateArray.shape[0]):
                    tempAddMoney = orderStateArray[i][2]
                    tempAddTime = orderStateArray[i][3]
                    reward = tempAddMoney * pow(0.98, tempAddTime)
                    rewardList.append(reward)
                action = rewardList.index(max(rewardList))

            else:
                action = agent.take_action(stateMatchArray)

            ## 记录此次派单方案
            stateMatchSolution = stateMatchArray  # array类型，n(骑手数目） * m + k + 200 （骑手特征+订单特征+供需特征）
            courierSolution = courierList[action]  # 选择哪个骑手
            addMoney = orderStateArray[action][2]
            addTime = orderStateArray[action][3]
            overdueSymbol = orderStateArray[action][6]
            reward = env.cal_reward(courierSolution, addMoney, addTime)
            d = DispatchSolution()
            d.add_state(stateMatchSolution)
            d.add_action(action)  # 标量
            d.add_reward(reward)  # 标量
            dDict[courierSolution] = d  # 记录骑手和d
            courierSolution.add_new_order(order)  # 骑手拿到新订单

        # 所有订单都被分配了骑手
        dispatchDict = env.step(dDict)  # 整个系统向前推进，同时返回完整的dDict
        T = T + 1

        stepReward = []
        for _, dispatch in dispatchDict.items():
            state, action, reward, nextState = process_memory(dispatch)
            replayBuffer.add(state, action, reward, nextState)
            stepReward.append(reward)
            dayReward.append(reward)
        meanStepReward = round(float(np.mean(np.array(stepReward))), 4)
        stepRecorder.write(str(meanStepReward) + "\n")
        #print(f'time_step{T}: {meanStepReward}.')

        if T == 179:
            for _ in range(20):
                batchState, batchAction, batchReward, batchNextState = replayBuffer.sample()
                agent.update(batchState, batchAction, batchReward, batchNextState)

    courierAccEfficiency = []
    for courier in env.courierList:
        courierAccEfficiency.append(courier.accEfficiency)
        env.overdueOrder += courier.route.overdueOrder
        realOrderTime += courier.accOrderTime
    fullOrder = 0
    for slotOrder in env.dayOrder:
        fullOrder += len(slotOrder)

    stepRecorder.close()
    '''
    for courier in env.courierList:
        for (money,time) in courier.route.realRewardList:
            reward = 10 * money * pow(0.95, time)
            realDayReward.append(reward)
    '''
    meanDayReward = round(float(np.mean(np.array(dayReward))), 4)
    dayRecorder.write('day' + str(dayIndex) + 'meanReward:' + str(meanDayReward) + '\n')
    dayRecorder.write('day' + str(dayIndex) + 'meanEff:' + str(np.mean(np.array(courierAccEfficiency))) + '\n')
    dayRecorder.write('day' + str(dayIndex) + 'varEff:' + str(np.std(np.array(courierAccEfficiency))) + '\n')
    dayRecorder.write('day' + str(dayIndex) + 'overdueRate:' + str(round(env.overdueOrder / fullOrder, 4)) + '\n')
    print(f'Day {dayIndex}: mean reward: {meanDayReward}.')
    print(f'Day {dayIndex}: mean efficiency for couriers: {np.mean(np.array(courierAccEfficiency))}.')
    print(f'Day{dayIndex}: mean overdue rate for couriers: {round(env.overdueOrder / fullOrder, 4)}.')
    print(f'Day{dayIndex}: Number of overdue orders in the day: {env.overdueOrder}.')
    print(f'Day{dayIndex}: Number of orders in the day: {fullOrder}.')
    #(f'Day{dayIndex}:{realOrderTime - originalOrderTime}.')
    #print(realOrderTime)
    #print(originalOrderTime)
    dayIndex += 1



dayRecorder.close()




#torch.save(agent,'../model/agent_seven(0.3,0.001).pth')








