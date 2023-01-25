from simulator.utility import cal_distance,cal_best_route,cal_route_dir
import copy



class Route:  # 设置路径规划

    def __init__(self,currentNode):
        self.orderList = [] # 骑手当前的订单列表
        self.orderFlag = {} # 骑手当前的订单状态 # 订单配送状态顺序  # 0:待去店面  1:在店面等餐  2:待送达  3:送达
        self.node = currentNode # 骑手当前的位置
        self.cityTime = None

        self.routeNodeList = []  # 订单节点顺序 (node,order)  # 到达一个地方，订单状态+1
        self.routeTimeList = []   # 每个路段需要花多长时间

        self.nextNode = None # 下一个去的位置
        self.nextOrder = None # 下一个订单
        self.nextPeriod = None # 去下一个位置剩下的时间
        self.state = None  # 0:待去店面  1:在店面等餐  2:待送达  3:送达

        self.remainPeriod = 0  # 格与格之间剩下的时间

        self.waitingPeriod = 3
        self.speedPeriod = 4
        self.routeMoney = 0  # 记录收益
        self.routeTime = 0 # 记录时间

        self.bestRoute = None # 当前最优的路径
        self.overdueOrder = 0

    def renew_order_list(self,orderList,orderFlag):  # 对齐骑手和route
        self.orderList = orderList
        self.orderFlag = orderFlag

    def add_new_order(self,order):
        self.orderList.append(order)
        self.orderFlag[order.orderID] = 0

    def delete_order(self,order): # 丢弃这个order
        for i in range(len(self.orderList)):
            if self.orderList[i].orderID == order.orderID:
                break
        self.orderList.pop(i)
        _ = self.orderFlag.pop(order.orderID)


    def route_generate(self):  # 生成订单顺序
        routeNodeList = []
        routeTimeList = []
        tempOrderFlag = copy.deepcopy(self.orderFlag)
        tempOrderList = copy.deepcopy(self.orderList)
        if self.state == 1: # 如果当前的状态为等餐,执行完这个状态
            routeNodeList.append((self.nextNode,self.nextOrder))
            routeTimeList.append(self.nextPeriod)
            tempOrderFlag[self.nextOrder.orderID] += 1

        pendingOrderList = []
        currentNode = self.node   # 骑手当前位置
        for orderID,flag in tempOrderFlag.items():
            newOrder = None
            for order in tempOrderList:
                if order.orderID == orderID:
                    newOrder = order
                    break
            if flag == 0:
                pendingOrderList.append((newOrder.orderMerchant,newOrder))
            elif flag == 2:
                pendingOrderList.append((newOrder.orderUser,newOrder))
            else:
                pass
        while (len(pendingOrderList) > 0 ):
            distanceList = []
            for (node,order) in pendingOrderList:
                distanceList.append(cal_distance(currentNode,node))
            minDistance = min(distanceList)
            node,order = pendingOrderList[distanceList.index(minDistance)]

            if tempOrderFlag[order.orderID] == 0:
                routeNodeList.append((node,order))
                routeTimeList.append(minDistance * self.speedPeriod)
                tempOrderFlag[order.orderID] += 1
                routeNodeList.append((node,order))
                routeTimeList.append(self.waitingPeriod)
                tempOrderFlag[order.orderID] += 1
            elif tempOrderFlag[order.orderID] == 2:
                routeNodeList.append((node,order))
                routeTimeList.append(minDistance * self.speedPeriod)
                tempOrderFlag[order.orderID] += 1
            else:
                pass

            currentNode = node  # 骑手新的位置
            pendingOrderList = []
            for orderID, flag in tempOrderFlag.items():
                newOrder = None
                for order in tempOrderList:
                    if order.orderID == orderID:
                        newOrder = order
                        break
                if flag == 0:
                    pendingOrderList.append((newOrder.orderMerchant, newOrder))
                elif flag == 2:
                    pendingOrderList.append((newOrder.orderUser, newOrder))
                else:
                    pass

        return routeNodeList,routeTimeList    # 订单节点顺序



    def route_update(self):  # 如果self.nextOrder == None 需要去巡游
        period = 0
        position = 0
        wholePeriod = 5

        if len(self.routeNodeList) == 0:   # 骑手现在没有订单配送
            self.nextNode = self.node
            self.nextOrder = None
            self.nextPeriod = 0
            self.state = 0
            return

        wholePeriod = wholePeriod - self.remainPeriod  # 上一步剩下的时间完成
        if self.remainPeriod > self.speedPeriod/2:
            currentNode = self.bestRoute[0]
            self.bestRoute = self.bestRoute[1:]
        else:
            currentNode = self.node  # 当前节点
        self.remainPeriod = 0

        # 这一步可以简化一些
        self.bestRoute,_ = cal_best_route(currentNode,self.routeNodeList[position][0])
        self.bestRoute = self.bestRoute[1:]


        while (wholePeriod > 0):
            if len(self.bestRoute) == 0: # 如果未来的路径为0,即骑手当前的节点位置已经到达该条路径的末端
                currentNode = self.routeNodeList[position][0]  # 前进到该位置
                self.orderFlag[self.routeNodeList[position][1].orderID] += 1  # 订单状态更新

                if self.orderFlag[self.routeNodeList[position][1].orderID] == 3:  # 如果订单完成将订单从骑手任务中去除
                    tempOrder = self.routeNodeList[position][1]
                    deliveryTime = (5 - wholePeriod) + (self.cityTime - tempOrder.orderCreateTime) * 5
                    if deliveryTime <= tempOrder.orderPromisePeriod:
                        addMoney = tempOrder.price
                    else:
                        self.overdueOrder += 1
                        addMoney = 0 * tempOrder.price
                    self.routeMoney += addMoney
                    self.routeTime += deliveryTime
                    self.delete_order(self.routeNodeList[position][1])

                position += 1
                if position == len(self.routeNodeList):  # 如果后续没有route了
                    self.nextNode = currentNode
                    self.nextOrder = None
                    self.nextPeriod = 0
                    self.state = 0
                    self.routeNodeList = []
                    self.routeTimeList = []
                    return
                self.bestRoute, _ = cal_best_route(currentNode, self.routeNodeList[position][0])
                self.bestRoute = self.bestRoute[1:]

                if self.orderFlag[self.routeNodeList[position][1].orderID] == 1:
                    if wholePeriod < self.waitingPeriod:
                        self.remainPeriod = self.waitingPeriod - wholePeriod
                    else:
                        self.remainPeriod = 0
                    wholePeriod -= self.waitingPeriod

            else: # 在路上
                if wholePeriod < self.speedPeriod:
                    if self.speedPeriod - wholePeriod > self.speedPeriod / 2:  # 没走过一半
                        self.remainPeriod = self.speedPeriod - wholePeriod
                    else:
                        currentNode = self.bestRoute[0]
                        self.bestRoute = self.bestRoute[1:]
                        self.remainPeriod = self.speedPeriod - wholePeriod
                else:
                    currentNode = self.bestRoute[0]
                    self.bestRoute = self.bestRoute[1:]
                    self.remainPeriod = 0
                wholePeriod -= self.speedPeriod


        self.node = currentNode
        self.nextNode = self.routeNodeList[position][0]
        self.nextOrder = self.routeNodeList[position][1]
        self.nextPeriod = self.remainPeriod
        self.state = self.orderFlag[self.routeNodeList[position][1].orderID]
        self.routeNodeList = self.routeNodeList[position:]
        usedPeriod = 5 - sum(self.routeTimeList[:position])
        self.routeTimeList = self.routeTimeList[position:]
        self.routeTimeList[0] = self.routeTimeList[0] - usedPeriod


















        # 必须完成上一步的单步任务
        '''


        period += self.routeTimeList[position]  # 加上下一个位置的时间
        self.bestRoute,_ = cal_best_route(currentNode,self.routeNodeList[position][0])
        if len(self.bestRoute) > 0:
            self.bestRoute = self.bestRoute[1:]

        while(period <= wholePeriod):  # 如果时间比剩下的时间少
            currentNode = self.routeNodeList[position][0]  # 前进到该位置
            self.orderFlag[self.routeNodeList[position][1]] += 1 # 订单状态更新
            if self.orderFlag[self.routeNodeList[position][1]] == 3:  # 如果订单完成将订单从骑手任务中去除
                self.delete_order(self.routeNodeList[position][1])
            position += 1
            if position == len(self.routeTimeList):  # 如果后续没有route了
                self.nextNode = currentNode
                self.nextOrder = None
                self.nextPeriod = 0
                self.state = 0
                break
            period += self.routeTimeList[position]
            self.bestRoute,_ = cal_best_route(currentNode,self.routeNodeList[position][0])
            if len(self.bestRoute) > 0:
                self.bestRoute = self.bestRoute[1:]


        self.nextNode = self.routeNodeList[position][0]  # 记录骑手下一站的位置
        self.state = self.orderFlag[self.routeNodeList[position][1]]  # 记录下下一站订单的状态
        if self.state == 1:  # 如果下一站的状态是等餐
            self.node = currentNode
            self.nextPeriod = self.waitingPeriod - (wholePeriod - (period - self.routeTimeList[position]))
            self.remainPeriod = 0  # 只用在前进一半的情况下
        else:
            self.nextPeriod = self.speedPeriod - (wholePeriod - (period - self.routeTimeList[position]))
            if self.nextPeriod >= 0 and self.nextPeriod < self.speedPeriod/2:  # 已经走过半
                self.node = self.bestRoute[0]  # 属于下一个node
                self.bestRoute = self.bestRoute[1:]  # 路径向前进一个
                self.remainPeriod = self.nextPeriod
            elif self.nextPeriod >= self.speedPeriod/2:
                self.node = currentNode
                self.remainPeriod = self.nextPeriod
            elif self.nextPeriod < 0:  # 仍然是属于下一个node
                self.node = self.bestRoute[0]
                self.bestRoute = self.bestRoute[1:]  # 路径向前进一个
                self.remainPeriod = self.speedPeriod + self.nextPeriod
            else:
                pass

            '''









