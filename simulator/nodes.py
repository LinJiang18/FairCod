


class Node(object):  # 结点应该怎么写
    def __init__(self,index):
        self.nodeIndex = index # 这个结点的index
        self.nodeCate = None   # 结点的类别

        # 时间信息
        self.cityDay = 0
        self.cityTime = 0

        # 骑手信息
        self.courierList = []  # 区域骑手列表(每个step都需要更新)
        self.courierDict = {}  # 区域字典

        #订单信息
        self.orderList = []  # 一个step内区域订单信息(每个step都需要更新)

        # 结点供需信息
        self.nodeSupply = 0  # 骑手的空间
        self.nodeDemand = 0 # 订单量

        # 邻居网格
        self.neighborNum = 0
        self.neighbors = []  # 周围六个邻居网格
        self.leftNeighbor = None # 正左边邻居
        self.bottomleftNeighbor = None # 左下方邻居
        self.upperleftNeighbor = None #左上方邻居
        self.rightNeighbor = None # 正右边邻居
        self.bottomrightNeighbor = None # 右下方邻居
        self.upperrightNeighbor = None # 右上方邻居

        # 最近的商业区
        self.nearCommerceArea = None  # 最近的商业区

    # 节点基本信息设置
    def set_node_cate(self,cate):
        self.nodeCate = cate

    def get_day_info(self,day):
        self.cityDay = day


    # 计算邻居
    def cal_neighbor(self):
        i = self.nodeIndex
        left = i - 1 if i % 10 != 0 else None
        right = i + 1 if i % 10 != 9 else None
        bottomleft = None
        upperleft = None
        bottomright = None
        upperright = None
        if int(i / 10) % 2 == 0:  # 为偶数，靠左边
            bottomleft = i + 10 - 1 if i % 10 != 0 and i + 10 <= 99 else None
            upperleft = i - 10 - 1 if i % 10 != 0 and i - 10 >= 10 else None
            bottomright = i + 10 if i + 10 <= 99 else None
            upperright = i - 10 if i - 10 >= 0 else None

        if int(i / 10) % 2 == 1:  # 为奇数，靠右边
            bottomleft = i + 10 if (i + 10) <= 99 else None
            upperleft = i - 10 if (i - 10) >= 0 else None
            bottomright = i + 10 + 1 if i % 10 != 9 and i + 10 <= 99 else None
            upperright = i - 10 + 1 if i % 10 != 9 and i - 10 >= 0 else None

        return [left, bottomleft, upperleft, right, bottomright, upperright]


    # 设置邻居
    def  set_neighbors(self):
        neighbors = self.cal_neighbor()
        self.neighbors = neighbors

    def set_single_neighbor(self):
        self.leftNeighbor = self.neighbors[0]  # 正左边邻居
        self.bottomleftNeighbor = self.neighbors[1]  # 左下方邻居
        self.upperleftNeighbor = self.neighbors[2]  # 左上方邻居
        self.rightNeighbor = self.neighbors[3]  # 正右边邻居
        self.bottomrightNeighbor = self.neighbors[4]  # 右下方邻居
        self.upperrightNeighbor = self.neighbors[5]  # 右上方邻居
        self.neighborNum = len(self.neighbors)

    def set_nearCommerceArea(self,index):  # 设置最近的商业区
        self.nearCommerceArea = index

    def clean_step_order_courier(self):
        self.orderList = []
        self.courierList = []


    def set_step_orderList(self,order):  # 每一步导入这个node中的订单
        self.orderList.append(order)

    def set_step_courierList(self,courier): # 每一步导入这个node中的骑手
        self.courierList.append(courier)







