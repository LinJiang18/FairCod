from route_planning.route import *
from simulator.utility import cal_distance,cal_best_route,cal_route_dir
import copy

class Courier(object):

    def __init__(self,courierID,courierInitNode):

        # 骑手基本属性
        self.courierID = courierID
        self.capacity = 6
        self.velocity = 4   # 从网格进入相邻网格需要四分钟
        self.waitingPeriod = 3 # 骑手在不同区域的等餐时间（分层）
        self.accPeriod = 0 # 骑手累计工作时间
        self.accMoney = 0 # 骑手累计收入
        self.accOrderTime = 0 # 累计订单时间
        self.accEfficiency = 0 # 骑手效率
        self.initNode = courierInitNode # 初始的节点位置

        #环境时间信息
        self.cityDay = 0   # 骑手当前所处城市天数
        self.cityTime =0   # 骑手当前所处城市时间 (180)

        #  骑手订单情况
        self.orderList = []  # 骑手拥有的订单信息
        self.orderFlag = {} # 骑手所有订单的配送状态
        self.orderNum = 0 # 骑手当前订单量

        # 骑手位置
        self.node = 0   # 骑手当前结点位置

        # 骑手配送路径
        self.route = None  # 赋值骑手路径信息



    def get_day_info(self,day):
        self.cityDay = day


    def set_loc(self,initNode):  # 将骑手当前位置设置为初始位置
        self.node = initNode

    def clean_order_info(self):
        self.orderList = []
        self.orderFlag = {}
        self.orderNum = 0
        self.route = Route(self.node)
        self.route.cityTime = self.cityTime
        self.route.orderList = self.orderList
        self.route.orderFlag = self.orderFlag

    def add_new_order(self,order):
        self.orderList.append(order)
        self.orderFlag[order.orderID] = 0
        self.orderNum = len(self.orderList)






