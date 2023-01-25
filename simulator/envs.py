from simulator.nodes import *
from simulator.couriers import *
from simulator.orders import *
from route_planning.route import *
import numpy as np
from simulator.utility import cal_distance,cal_best_route,cal_route_dir
import pandas as pd

class Region:

    def __init__(self,courierInit,realOrders,M,N,maxDay,maxTime):
        self.courierInit = courierInit  # 骑手信息初始化 (设置多少个骑手)

        # 全体订单信息：
        self.wholeOrder = realOrders  # 订单信息

        # 全体时间信息
        self.cityDay = 0  # 对应系统发展的第几天
        self.cityTime = 0 # 对应这一天的什么时候
        self.maxCityTime = maxTime # 从早上8点到晚上10点，共计统计15个小时
        self.maxCityDay = maxDay # 最大的城市天数

        # 全体骑手信息
        self.courierList = []  # 骑手列表
        self.courierDict = {}  # 骑手ID对应的字典
        self.courierNum = 0  # 对应的总骑手数目
        self.nodeCourier = [] # 所有区域的实时骑手信息，从0~99

        # 全体网格信息
        self.M = M  # 结点横轴数
        self.N = N  # 结点纵轴数
        self.nodeNum = self.M * self.N # 总的结点数
        self.nodeList = [Node(i) for i in range(self.M * self.N)]  # 区域的结点列表
        self.nodeDict = {}   # nodeIndex与node的配对，与nodeList一致


        # 结点类别信息
        self.commerceAreaList = []  # 商业区
        self.foodAreaList = []  # 饮食区
        self.residentAreaList = []  # 居民区
        self.suburbAreaList = []  # 郊区

        # 时间步信息
        self.dayOrder = []  # 这一天所有的订单信息，按0~179进行排列

        # 一些语境信息
        self.dispatchRegion = 3  # 派单范围
        self.trajectoryStateLength = 20  # 路径特征长度
        self.maxOrderNum = 4 # 骑手能携带的最大订单数

        # 公平信息
        self.meanEfficiency = 0
        self.alpha = 1
        self.beta = 0
        self.overdueOrder = 0


    def set_node_info(self):
        # 设置结点的类别

        # 结点类别矩阵
        cateMatrix = [[3, 3, 2, 2, 1, 3, 3, 3, 2, 3],
                      [3, 3, 1, 2, 3, 3, 3, 1, 1, 2],
                      [2, 0, 2, 3, 3, 3, 3, 0, 1, 2],
                      [3, 2, 1, 2, 2, 1, 3, 2, 3, 3],
                      [3, 3, 1, 0, 3, 1, 2, 2, 1, 3],
                      [1, 3, 3, 2, 2, 0, 0, 2, 3, 3],
                      [3, 3, 2, 0, 2, 2, 2, 0, 1, 2],
                      [3, 1, 2, 3, 3, 3, 3, 2, 2, 3],
                      [3, 1, 2, 2, 3, 3, 3, 2, 0, 2],
                      [3, 3, 2, 0, 3, 3, 3, 1, 2, 1]]   # 根据真实场景自己设置地图
        cateMatrix = list(np.array(cateMatrix).reshape(100))

        # 设置类别
        for i in range(self.nodeNum):
            self.nodeList[i].set_node_cate(3-cateMatrix[i]) # 给每个节点分配cate
            # 3 商业区 /  2 饮食区   / 1 居民区  / 0 郊区

        # 将节点的类别分配到region的节点类别信息中
        for node in self.nodeList:
            if node.nodeCate == 3:
                self.commerceAreaList.append(node.nodeIndex)
            elif node.nodeCate == 2:
                self.foodAreaList.append(node.nodeIndex)
            elif node.nodeCate == 1:
                self.residentAreaList.append(node.nodeIndex)
            elif node.nodeCate == 0:
                self.suburbAreaList.append(node.nodeIndex)
            else:
                pass

        # 设置邻居节点
        for node in self.nodeList:
            node.set_neighbors()   # 给node赋上邻居结点

        self.cal_node_neighbors()



        # 设置距离最近的商业区:
        for node in self.nodeList:
            disList = []
            for commerceNodeIndex in self.commerceAreaList:
                dis = cal_distance(node.nodeIndex,commerceNodeIndex)  # 计算node距离商业区的距离
                disList.append((commerceNodeIndex,dis))
            nearCommerceAreaIndex = min(disList,key= lambda x:x[1])[0]   # 选择最近的那一个
            node.set_nearCommerceArea(nearCommerceAreaIndex)

        # 将nodeList赋值到nodeDict
        for node in self.nodeList:
            self.nodeDict[node.nodeIndex] = node

    def cal_node_neighbors(self):
        for node in self.nodeList:
            nodeNeighbors = []
            for num in node.neighbors:
                if num is not None:
                    nodeNeighbors.append(self.nodeList[num])
                else:
                    nodeNeighbors.append(None)
            node.neighbors = nodeNeighbors
            node.set_single_neighbor()

    def set_courier_info(self):   # 设置节点的初始信息,包含骑手ID和骑手初始位置
        for courierInfo in self.courierInit:
            self.courierList.append(Courier(courierInfo[0],courierInfo[1]))
        self.courierNum = len(self.courierList)
        for courier in self.courierList:
            self.courierDict[courier.courierID] = courier

    def get_day_info(self,day):
        self.cityDay = day
        for courier in self.courierList:
            courier.get_day_info(day)
        for node in self.nodeList:
            node.get_day_info(day)

    def reset_clean(self):
        self.cityTime = 0  #    城市时间设置为0
        self.overdueOrder = 0 # 上一天的超时订单清0
        self.reset_courier_info()  # 重设骑手初始位置,重置骑手信息
        self.bootstrap_one_day_order()  # 导入这一天的订单
        self.clean_step_order_courier()   # 清楚上一步的订单和骑手信息
        self.step_bootstrap_order()  # 按时间将订单导入节点中
        self.step_bootstrap_courier()  # 按时间将骑手导入节点中

    def reset_courier_info(self):
        for courier in self.courierList:
            courier.cityTime = 0
            courier.accOrderTime = 0
            courier.set_loc(courier.initNode)   # 设置骑手的位置
            courier.clean_order_info()


    def bootstrap_one_day_order(self):
        dayOrder = [[] for _ in np.arange(self.maxCityTime)]  # 180个
        for dayWholeOrder in self.wholeOrder[self.cityDay]:  # 循环这一天
            for order in dayWholeOrder:  # 循环这一天每个时间的订单
                # 订单信息包含：ID,day,time,merchantNode,userNode,distance,promiseTime,price
                startTime = int(order[2])  # 订单时间
                dayOrder[startTime].append(Order(order[0], order[1], order[2], order[3], order[4], order[5], order[6], order[7]))
        self.dayOrder = dayOrder  # 表示day_orders


    def clean_step_order_courier(self):
        for node in self.nodeList:
            node.clean_step_order_courier()

    def step_bootstrap_order(self):  # 将这一步的订单放到对应的每个网格中
        timeOrder = self.dayOrder[self.cityTime]
        for order in timeOrder:
            self.nodeList[order.orderMerchant].set_step_orderList(order)   # 将新订单放到每个对应的节点中

    def step_bootstrap_courier(self): # 将这一步的骑手放到对应的每个网格中
        for courier in self.courierList:
            self.nodeList[courier.node].set_step_courierList(courier)

    def action_collect(self,theOrder):   # 返回适合的骑手
        dispatchRegion = self.dispatchRegion
        pendingCourier = []
        newPendingCourier = []
        pendingNode = []
        while(len(newPendingCourier) == 0 and dispatchRegion <= 14):  # 如果最小范围内没有找到骑手，那就扩大范围找
            orderNode = theOrder.orderMerchant
            for i in range(self.nodeNum):
                if cal_distance(orderNode,i) <= dispatchRegion:
                    pendingNode.append(i)
            for node in pendingNode:
                for courier in self.nodeList[node].courierList:
                    pendingCourier.append(courier)
            newPendingCourier = []
            courierOverdueSymbol = 0
            for courier in pendingCourier:
                courier.route.add_new_order(theOrder)
                routeNodeList, routeTimeList = courier.route.routeNodeList,courier.route.routeTimeList
                for order in courier.orderList:   # 如果当前有超时
                    minIndex = len(routeNodeList) - 1
                    for i in range(len(routeNodeList) - 1, -1, -1):
                        if order.orderID == routeNodeList[i][1].orderID:
                            minIndex = i
                            break
                    deliveryTime = sum(routeTimeList[:minIndex + 1]) + (
                            courier.cityTime - order.orderCreateTime) * 5
                    if deliveryTime > order.orderPromisePeriod:
                        courierOverdueSymbol = 1
                        break
                if courier.orderNum < self.maxOrderNum and (courierOverdueSymbol & 1) == 0:
                    newPendingCourier.append(courier)
                courier.route.delete_order(theOrder)
                courierOverdueSymbol = 0
            dispatchRegion += 1
            pendingCourier = []


        return newPendingCourier


    def courier_state_compute(self,courierList): # 时间,骑手的位置,订单数,骑手的未来路径
        courierStateList = []
        for courier in courierList:
            time = courier.cityTime
            node = courier.node
            orderNum = len(courier.orderList)

            # 计算路径编码
            route = courier.route.routeNodeList
            routeNodeList = []  # 首先记录骑手当前位置
            for node, _ in route:
                routeNodeList.append(node)
            trajectoryList = []
            beginNode = courier.node
            for i in range(len(routeNodeList)):
                oneTrajectory,_ = cal_best_route(beginNode,routeNodeList[i])
                if i < len(routeNodeList) - 1:
                    oneTrajectory = oneTrajectory[:-1]  # 保留开头去掉末尾
                beginNode = routeNodeList[i]
                trajectoryList.append(oneTrajectory)
            trajectoryList = [x for y in trajectoryList for x in y]


            if len(trajectoryList) == 0:
                trajectoryList = [courier.node] * self.trajectoryStateLength
            elif len(trajectoryList) < self.trajectoryStateLength:  # 如果长度小于20
                trajectoryList += [trajectoryList[-1]] * (self.trajectoryStateLength - len(trajectoryList))
            elif len(trajectoryList) > self.trajectoryStateLength:  # 如果长度大于20
                removeNum = len(trajectoryList) - self.trajectoryStateLength
                intervalNum = int(len(trajectoryList) / removeNum)
                removeIndex = intervalNum - 1
                for i in range(removeNum):
                    trajectoryList.pop(removeIndex)
                    removeIndex += (intervalNum - 1)
            else:
                pass

            courierState = [time,node,orderNum] + trajectoryList
            courierStateList.append(courierState)

        courierStateList = np.array(courierStateList)

        return courierStateList


    def order_state_compute(self,courierList,theOrder):
        # 订单起点，订单终点，订单增加收入，订单增加时间,骑手订单距离
        orderStateList = []
        orderMerchant = theOrder.orderMerchant
        orderUser = theOrder.orderUser
        overdueSymbol = 0  # 记录超时信息
        for theCourier in courierList:
            routeNodeList, routeTimeList = theCourier.route.routeNodeList,theCourier.route.routeTimeList
            oriAddTime = sum(routeTimeList)
            oriAddMoneyList = {}
            for order in theCourier.orderList:
                minIndex = len(routeNodeList) - 1
                for i in range(len(routeNodeList) - 1, -1, -1):
                    if order.orderID == routeNodeList[i][1].orderID:
                        minIndex = i
                        break
                deliveryTime = sum(routeTimeList[:minIndex + 1]) + (theCourier.cityTime - order.orderCreateTime) * 5
                if deliveryTime <= order.orderPromisePeriod:
                    addMoney = order.price
                else:
                    addMoney = 0 * order.price
                oriAddMoneyList[order.orderID] = addMoney



            theCourier.route.add_new_order(theOrder)
            routeNodeList, routeTimeList = theCourier.route.route_generate()
            afterAddTime = sum(routeTimeList)
            afterAddMoneyList = {}
            for order in theCourier.orderList:
                minIndex = len(routeNodeList) - 1
                for i in range(len(routeNodeList) - 1, -1, -1):
                    if order.orderID == routeNodeList[i][1].orderID:
                        minIndex = i
                        break
                deliveryTime = sum(routeTimeList[:minIndex + 1]) + (theCourier.cityTime - order.orderCreateTime) * 5
                if deliveryTime <= order.orderPromisePeriod:
                    addMoney = order.price
                else:
                    overdueSymbol += 1
                    addMoney = 0 * order.price
                afterAddMoneyList[order.orderID] = addMoney
            theCourier.route.delete_order(theOrder)

            absoluteValue = 0
            for order in theCourier.orderList:
                absoluteValue += (afterAddMoneyList[order.orderID] - oriAddMoneyList[order.orderID])
            absoluteValue += afterAddMoneyList[theOrder.orderID]
            addTime = afterAddTime - oriAddTime
            addMoney = absoluteValue
            promisePeriod = theOrder.orderPromisePeriod

            distance = cal_distance(theCourier.node, theOrder.orderMerchant)
            if overdueSymbol >= 1:
                overdueSymbol = 1
            orderState = [orderMerchant,orderUser,addMoney,addTime,distance,promisePeriod,overdueSymbol]
            orderStateList.append(orderState)
            overdueSymbol = 0
        orderStateList = np.array(orderStateList)
        return orderStateList


    def sd_state_compute(self):
        sdStateList = []
        for node in self.nodeList:
            nodeSupply = len(node.courierList) # 供应是节点内的骑手数目
            nodeDemand = len(node.orderList)  # 需要是节点内的订单数目
            sdStateList.append(nodeSupply)
            sdStateList.append(nodeDemand)
        return np.array(sdStateList)

    def cal_reward(self,theCourier,addMoney,addTime,overdueSymbol):
        self.cal_courier_efficiency()  # 更新骑手的效率,并计算平均效率
        if addMoney < 0:
            rewardOne = 0
        else:
            rewardOne = addMoney * pow(0.98,addTime)
        rewardTwo = theCourier.accEfficiency - self.meanEfficiency
        reward = self.alpha * rewardOne + self.beta * (-rewardTwo)
        return  reward

    def cal_courier_efficiency(self):
        accEfficiency = 0
        for courier in self.courierList:
            if courier.accPeriod == 0:
                courier.accEfficiency = 0
            else:
                courier.accEfficiency = courier.accMoney / (courier.accPeriod/60)  # 收入/工作时长()
            accEfficiency += courier.accEfficiency
        self.meanEfficiency = accEfficiency / len(self.courierList)

    def update_time(self):
        self.cityTime += 1
        for courier in self.courierList:
            courier.cityTime += 1
            courier.route.cityTime += 1
        for node in self.nodeList:
            node.cityTime += 1

    def step(self,dDict): # 时间向前推进
        self.update_time()  # 更新节点和骑手的更新信息
        for courier in self.courierList: # 对于courierList中的骑手更新
            courier.route.renew_order_list(courier.orderList,courier.orderFlag)
            courier.route.routeNodeList,courier.route.routeTimeList = courier.route.route_generate()
            courier.route.route_update()
            courier.orderList = courier.route.orderList
            courier.orderFlag = courier.route.orderFlag
            courier.node = courier.route.node
            courier.orderNum = len(courier.orderList)
            courier.accPeriod += 5
            courier.accMoney += courier.route.routeMoney
            courier.accOrderTime += courier.route.routeTime
            courier.route.routeMoney = 0
            courier.route.routeTime = 0


        # 更新骑手在节点中的位置,导入新订单
        self.clean_step_order_courier()
        if self.cityTime < 180:
            self.step_bootstrap_order()
            self.step_bootstrap_courier()

        # 计算新的供需
        supplydemandStateArray = self.sd_state_compute()  # np.array类型  1 * 200

        # 记录next_state
        dispatchDict = self.cal_next_state(dDict,supplydemandStateArray)

        return dispatchDict


    def cal_next_state(self,dDict,supplydemandStateArray):
        dispatchDict = {}
        for courier, d in dDict.items():  # 时间,骑手的位置,订单数,骑手的未来路径,未来订单
            time = courier.cityTime
            node = courier.node
            orderNum = len(courier.orderList)
            route = courier.route.routeNodeList
            routeNodeList = []  # 首先记录骑手当前位置
            for node, _ in route:
                routeNodeList.append(node)
            trajectoryList = []
            beginNode = courier.node
            for i in range(len(routeNodeList)):
                oneTrajectory, _ = cal_best_route(beginNode, routeNodeList[i])
                if i < len(routeNodeList) - 1:
                    oneTrajectory = oneTrajectory[:-1]  # 保留开头去掉末尾
                beginNode = routeNodeList[i]
                trajectoryList.append(oneTrajectory)
            trajectoryList = [x for y in trajectoryList for x in y]

            if len(trajectoryList) == 0:
                trajectoryList = [courier.node] * self.trajectoryStateLength
            if len(trajectoryList) < self.trajectoryStateLength:  # 如果长度小于20
                trajectoryList += [trajectoryList[-1]] * (self.trajectoryStateLength - len(trajectoryList))
            elif len(trajectoryList) > self.trajectoryStateLength:  # 如果长度大于20
                removeNum = len(trajectoryList) - self.trajectoryStateLength
                intervalNum = int(len(trajectoryList) / removeNum)
                removeIndex = intervalNum - 1
                for i in range(removeNum):
                    trajectoryList.pop(removeIndex)
                    removeIndex += (intervalNum - 1)
            else:
                pass

            nearOrderList = []
            dispatchRegion = self.dispatchRegion
            while len(nearOrderList) == 0:
                for order in self.dayOrder[self.cityTime]:
                    if cal_distance(order.orderMerchant, courier.node) <= dispatchRegion:
                        nearOrderList.append(order)
                dispatchRegion += 1
            rewardList = []

            routeNodeList, routeTimeList = courier.route.routeNodeList, courier.route.routeTimeList
            oriAddTime = sum(routeTimeList)
            oriAddMoneyList = {}
            for order in courier.orderList:
                minIndex = len(routeNodeList) - 1
                for i in range(len(routeNodeList) - 1, -1, -1):
                    if order.orderID == routeNodeList[i][1].orderID:
                        minIndex = i
                        break
                deliveryTime = sum(routeTimeList[:minIndex + 1]) + (
                        courier.cityTime - order.orderCreateTime) * 5
                if deliveryTime <= order.orderPromisePeriod:
                    addMoney = order.price
                else:
                    addMoney = 0 * order.price
                oriAddMoneyList[order.orderID] = addMoney

            for theOrder in nearOrderList:
                overdueSymbol = 0
                courier.route.add_new_order(theOrder)
                routeNodeList, routeTimeList = courier.route.route_generate()
                afterAddTime = sum(routeTimeList)
                afterAddMoneyList = {}
                for order in courier.orderList:
                    minIndex = len(routeNodeList) - 1
                    for i in range(len(routeNodeList) - 1, -1, -1):
                        if order.orderID == routeNodeList[i][1].orderID:
                            minIndex = i
                            break
                    deliveryTime = sum(routeTimeList[:minIndex + 1]) + (
                            courier.cityTime - order.orderCreateTime) * 5
                    if deliveryTime <= order.orderPromisePeriod:
                        addMoney = order.price
                    else:
                        overdueSymbol += 1
                        addMoney = 0 * order.price
                    afterAddMoneyList[order.orderID] = addMoney
                courier.route.delete_order(theOrder)

                absoluteValue = 0
                for order in courier.orderList:
                    absoluteValue += (afterAddMoneyList[order.orderID] - oriAddMoneyList[order.orderID])
                absoluteValue += afterAddMoneyList[theOrder.orderID]
                addTime = afterAddTime - oriAddTime
                addMoney = absoluteValue
                if overdueSymbol >= 1:
                    overdueSymbol = 1
                reward = addMoney * pow(0.98,addTime) * (1 - overdueSymbol)
                rewardList.append(reward)

            orderIndex = rewardList.index(max(rewardList))
            theOrder = nearOrderList[orderIndex]

            overdueSymbol = 0
            courier.route.add_new_order(theOrder)
            routeNodeList, routeTimeList = courier.route.route_generate()
            afterAddTime = sum(routeTimeList)
            afterAddMoneyList = {}
            for order in courier.orderList:
                minIndex = len(routeNodeList) - 1
                for i in range(len(routeNodeList) - 1, -1, -1):
                    if order.orderID == routeNodeList[i][1].orderID:
                        minIndex = i
                        break
                deliveryTime = sum(routeTimeList[:minIndex + 1]) + (
                        courier.cityTime - order.orderCreateTime) * 5
                if deliveryTime <= order.orderPromisePeriod:
                    addMoney = order.price
                else:
                    overdueSymbol += 1
                    addMoney = 0 * order.price
                afterAddMoneyList[order.orderID] = addMoney
            courier.route.delete_order(theOrder)

            absoluteValue = 0
            for order in courier.orderList:
                absoluteValue += (afterAddMoneyList[order.orderID] - oriAddMoneyList[order.orderID])
            absoluteValue += afterAddMoneyList[theOrder.orderID]
            addTime = afterAddTime - oriAddTime
            addMoney = absoluteValue
            if overdueSymbol >= 1:
                overdueSymbol = 1
            distance = cal_distance(theOrder.orderMerchant,courier.node)
            promisePeriod = theOrder.orderPromisePeriod
            nextActionState = [theOrder.orderMerchant,theOrder.orderUser,addMoney,addTime,distance,promisePeriod,overdueSymbol]
            nextState = [time, node, orderNum] + trajectoryList + list(supplydemandStateArray) + nextActionState
            nextState = np.array(nextState)
            d.add_nextState(nextState)
            dispatchDict[courier] = d

        return dispatchDict






















