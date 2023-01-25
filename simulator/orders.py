from simulator.utility import cal_distance,cal_best_route,cal_route_dir
#  ID,day,time,merchantNode,userNode,distance,promiseTime,price


class Order(object):
    def __init__(self,orderID,orderDay,orderTime,orderMerchant,orderUser,orderDistance,orderPromisePeriod,orderPrice):
        self.orderID = orderID
        self.orderDay = orderDay
        self.orderCreateTime = orderTime  # 哪个时间点创建了订单
        self.orderFinishTime = None
        self.orderMerchant = orderMerchant  # 商家的位置
        self.orderUser = orderUser   # 用户的位置
        self.orderDistance = orderDistance # 从商家到用户的位置
        self.price = orderPrice  # 订单价格
        self.orderPromisePeriod = orderPromisePeriod # 订单预计送达时间
        self.flag = 0 # flag表示还未取餐，1表示已经取餐但还未送达，后续在step中更新
        self.courierID = 0 # 订单被分配给哪个骑手
        self.realOrderTime = 0
        self.realOrderMoney = 0