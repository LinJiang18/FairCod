from simulator.nodes import Node
from simulator.dispatch import DispatchSolution
import numpy as np

M = 10
N = 10
nodeNum = 100
nodeList = [Node(i) for i in range(M * N)]
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
for i in range(nodeNum):
    nodeList[i].set_node_cate(3-cateMatrix[i]) # 给每个节点分配cate
    # 3 商业区 /  2 饮食区   / 1 居民区  / 0 郊区

# 设置邻居节点
for node in nodeList:
    node.set_neighbors()   # 给node赋上邻居结点

for node in nodeList:
    nodeNeighbors = []
    for num in node.neighbors:
        if num is not None:
            nodeNeighbors.append(nodeList[num])
        else:
            nodeNeighbors.append(None)
    node.neighbors = nodeNeighbors
    node.set_single_neighbor()


def cal_distance(oriIndex,destIndex):
    oriY = int(oriIndex /10)
    oriX = oriIndex % 10
    destY = int(destIndex /10)
    destX = destIndex % 10

    # 看纵向的差距
    distY = abs(oriY - destY)
    distX = abs(oriX- destX)
    if distY % 2 == 0:
        distance = distY if distY >= 2 * distX else distX + ( 1 /2) * distY
    else:
        if distY >= (2 * distX + 1):
            distance =distY
        else:
            distX = destX - oriX
            if distX > 0:  # 向右推进
                if destY % 2 == 0:  # 如果目的地是偶数，即从奇到偶
                    distance = (1 / 2) * (distY - 1) + distX
                else:
                    distance = (1 / 2) * (distY - 1) + distX + 1
            else:  # 向左推进
                if destY % 2 == 0:  # 如果目的地是偶数，即从奇到偶
                    distance = (1 / 2) * (distY - 1) + (-distX) + 1
                else:
                    distance = (1 / 2) * (distY - 1) + (-distX)

    return int(distance)


def cal_route_dir(oriIndex, destIndex):
    oriY = int(oriIndex / 10)
    oriX = oriIndex % 10
    destY = int(destIndex / 10)
    destX = destIndex % 10
    numOne = 0
    numTwo = 0
    numOneDir = 0
    numTwoDir = 0
    #   2    5

    # 0         3

    #   1    4

    # 建立两条对角线(确立方向和方向距离)

    if destY < oriY:  # 左上右上对角线
        # 起点对角线与终点所在行相交的左点
        upperleftDist = (oriY - destY) / 2 if (oriY - destY) % 2 == 0 else (oriY - destY - 1) / 2 if destY % 2 == 0 else\
            (oriY - destY - 1) / 2 + 1
        # 起点对角线与终点所在行相交的右点
        upperrightDist = (oriY - destY) / 2 if (oriY - destY) % 2 == 0 else (oriY - destY - 1) / 2 + 1 if destY % 2 == 0\
            else (oriY - destY - 1) / 2
        upperleftGridIndex = destY * 10 + oriX - upperleftDist
        upperrightGridIndex = destY * 10 + oriX + upperrightDist
        # 判断属于哪种情况
        if upperleftGridIndex >= destIndex:  # 左+左上
            numOne = upperleftGridIndex - destIndex
            numOneDir = 0
            numTwo = oriY - destY
            numTwoDir = 2
        elif upperleftGridIndex < destIndex and upperrightGridIndex > destIndex:  # 左上 + 右上
            numOne = destIndex - upperleftGridIndex
            numOneDir = 5
            numTwo = upperrightGridIndex - destIndex
            numTwoDir = 2
        elif upperrightGridIndex <= destIndex:  # 右 + 右上
            numOne = destIndex - upperrightGridIndex
            numOneDir = 3
            numTwo = oriY - destY
            numTwoDir = 5
        else:
            pass

    elif destY > oriY:  # 左下右下对角线
        # 起点对角线与终点所在行相交的左点
        bottomleftDist = (destY - oriY) / 2 if (destY - oriY) % 2 == 0 else (destY - oriY - 1) / 2 \
            if destY % 2 == 0 else (destY - oriY - 1) / 2 + 1
        # 起点对角线与终点所在行相交的右点
        bottomrightDist = (destY - oriY) / 2 if (destY - oriY) % 2 == 0 else (destY - oriY - 1) / 2 + 1 \
            if destY % 2 == 0 else (destY - oriY - 1) / 2
        bottomleftGridIndex = destY * 10 + oriX - bottomleftDist
        bottomrightGridIndex = destY * 10 + oriX + bottomrightDist

        # 判断属于哪种情况
        if bottomleftGridIndex >= destIndex:  # 左 + 左下
            numOne = bottomleftGridIndex - destIndex
            numOneDir = 0
            numTwo = destY - oriY
            numTwoDir = 1

        elif bottomleftGridIndex < destIndex and bottomrightGridIndex > destIndex:  # 左下 + 右下
            numOne = destIndex - bottomleftGridIndex
            numOneDir = 4
            numTwo = bottomrightGridIndex - destIndex
            numTwoDir = 1

        elif bottomrightGridIndex <= destIndex:  # 右 + 右下
            numOne = destIndex - bottomrightGridIndex
            numOneDir = 3
            numTwo = destY - oriY
            numTwoDir = 4

    else:
        if oriX > destX:
            numOne = oriX - destX
            numOneDir = 0
        elif oriX < destX:
            numOne = destX - oriX
            numOneDir = 3
        else:
            numOne = 0
            numOneDir = 0

        numTwo = 0
        numTwoDir = 0

    numOne = int(numOne)
    numTwo = int(numTwo)

    # dirDict = {0:'左',1:'左下',2:'左上',3:'右',5:'右上',4:'右下'}

    # print('first direction:' +str(dirDict[numOneDir]))
    # print('first direction distance:' + str(numOne))

    # print('second direction:' +str(dirDict[numTwoDir]))
    # print('second direction distance:' + str(numTwo))

    return numOne, numOneDir, numTwo, numTwoDir


def cal_best_route(oriIndex, destIndex):
    numOne, numOneDir, numTwo, numTwoDir = cal_route_dir(oriIndex, destIndex)
    stack = []
    oriNode = nodeList[oriIndex]
    destNode = nodeList[destIndex]

    # 遍历建树

    class TreeNode():

        def __init__(self, node, index, value, numOne, numTwo):
            self.node = node
            self.index = index
            self.value = value
            self.numOne = numOne
            self.numTwo = numTwo
            self.exploreSymbol = 0

            self.leftTreeNode = None
            self.rightTreeNode = None

        def add_leftNode(self, leftTreeNode):
            self.leftTreeNode = leftTreeNode

        def add_rightNode(self, rightTreeNode):
            self.rightTreeNode = rightTreeNode

        def set_exploreSymbol(self, symbol):
            self.exploreSymbol = symbol

    oriTreeNode = TreeNode(oriNode, oriNode.nodeIndex, oriNode.nodeCate, numOne, numTwo)
    stack = [oriTreeNode]
    nextStack = []
    while (len(stack) != 0):
        for treeNode in stack:
            if treeNode.numOne > 0:
                tempLeftNode = treeNode.node.neighbors[numOneDir]  # 提取节点在该方向的邻居
                if tempLeftNode is not None:
                    leftTreeNode = TreeNode(tempLeftNode, tempLeftNode.nodeIndex, tempLeftNode.nodeCate, treeNode.numOne - 1,
                                            treeNode.numTwo)
                    treeNode.add_leftNode(leftTreeNode)
                    nextStack.append(leftTreeNode)
                else:
                    pass
            else:
                pass

            if treeNode.numTwo > 0:
                tempRightNode = treeNode.node.neighbors[numTwoDir]  # 提取节点在该方向的邻居
                if tempRightNode is not None:
                    rightTreeNode = TreeNode(tempRightNode, tempRightNode.nodeIndex, tempRightNode.nodeCate, treeNode.numOne,
                                             treeNode.numTwo - 1)
                    treeNode.add_rightNode(rightTreeNode)
                    nextStack.append(rightTreeNode)
                else:
                    pass
            else:
                pass

        stack = [s for s in nextStack]
        nextStack = []

    # 遍历树找最优路径

    stack = []
    route = []
    value = oriTreeNode.value
    bestRoute = []
    bestValue = oriTreeNode.value
    stack.append(oriTreeNode)
    route.append(oriTreeNode.index)
    pivot = oriTreeNode

    while (len(stack) != 0):
        if pivot.exploreSymbol == 0:  # 探索左子树
            if pivot.leftTreeNode is not None:  # 修改状态并进入左子树
                stack.append(pivot.leftTreeNode)
                route.append(pivot.leftTreeNode.index)
                value += pivot.leftTreeNode.value
                pivot.exploreSymbol = 1
                pivot = pivot.leftTreeNode

            else:  # 修改状态
                pivot.exploreSymbol = 1
            continue



        elif pivot.exploreSymbol == 1:  # 探索右子树
            if pivot.rightTreeNode is not None:  # 修改状态并进入右子树
                stack.append(pivot.rightTreeNode)
                route.append(pivot.rightTreeNode.index)
                value += pivot.rightTreeNode.value
                pivot.exploreSymbol = 2
                pivot = pivot.rightTreeNode

            else:
                pivot.exploreSymbol = 2
            continue


        else:
            if pivot.index == destIndex:
                if value >= bestValue:
                    bestRoute = [r for r in route]
                    bestValue = value
            value -= stack.pop().value
            route.pop()
            if len(stack) == 0:
                break
            pivot = stack[-1]

    return bestRoute, bestValue


def process_memory(d):
    state = d.state
    action = d.action
    reward = d.reward
    nextState = d.nextState
    return state,action,reward,nextState