####################################################################################################################
#####################################  Simulation Project - Part B #################################################
################################ Tal Eylon, Amihai Kalev, Avihoo Menahem ###########################################
########################################  THIRD FORMATION ######################################################
####################################################################################################################

#################################################################################################
########################################## IMPORTS ##############################################
#################################################################################################
import math
import simpy
from simpy.util import start_delayed # delay the start of functions in the simulation
import numpy as np
import matplotlib.pyplot as plt # for graphs
plt.style.use("ggplot")

################################################################################################
########################################## POLICY ##############################################
################################################################################################
POLICY = 1 # the chosen policy to be used in this formation. do not change
np.random.seed(60)

#################################################################################################
########################################## CLASSES ##############################################
#################################################################################################

###############################################
######### Drone Station
###############################################
class DroneStation():
    """
    This class defines a drone station (hub). Each drone station has 3 operation centers:
    1 - basic drones, 2 - regular drones, 3 - super drones.
    """

    def __init__(self, env, location):
        self.env = env                        # define enviornment.
        self.location = location              # The point (X,Y) in which the drone station is located.

        # operation centers: the first - for basic drones, the second - for regular drones, the third - for super drones.
        self.operation_centers = [simpy.Resource(env, capacity=20), simpy.Resource(env,capacity=7), simpy.Resource(env,capacity=3)]



###############################################
######## Order
###############################################
class Order():
    """
    The order class. Contains all the relevant information of an order.
    """
    def __init__(self, env, destination):
        self.env = env                                        # the relevant simulation environment.
        self.destination = destination                        # The destination point (X,Y) in meters.
        self.delivered = 0                                    # at first, the order has not yet delivered
        self.delivered_in_another_day = 0                     # at first we assume the order has been delivered in the same day
        self.hub_location, self.distance_from_hub = 0,0       # at first, hub location is 0, distance is 0.
        self.size = self.raffleSize()                         # raffle the order's size - can be either 1, 2 or 3.
        self.drone_type = 0                                   # drone type, 0 at start. can be either 1, 2 or 3.
        self.service_time,self.one_way_time  = 0,0            # Service time at destination & the time it takes to go in
        # one way between the hub and the destination
        self.cost = {"distance": 0,"service": 0, "coupon": 0} # relevant order's costs.
        self.coupon_histogram = [0,0,0,0]
        self.order_times = [0, 0]        # first location - when the order accepted, second - when the order delivered.

    def raffleSize(self):
        """
        This function returns the size of the order's package.
        :return:
        """
        random_size = np.random.uniform(0, 1)
        if random_size < 0.1:
            size = 3
        elif random_size < 0.4:
            size = 2
        else:
            size = 1
        return size

    def updateServiceAndDistanceCost(self):
        """
        According to the service and distance costs, calculate the relevant cost.
        :return:
        """
        if self.drone_type == 1:
            cost_per_minute = 100/60
        elif self.drone_type == 2:
            cost_per_minute = 130/60
        else:
            cost_per_minute = 180/60

        self.cost["service"] = (2*self.one_way_time + self.service_time)*cost_per_minute
        self.cost["distance"] = (0.0001) * 2 * self.distance_from_hub

    def coupon(self):
        """
        for each order that takes longer then 2 hours, this function calculates the relevant penalty.
        :return:
        """
        total_time = (self.order_times[1]-self.order_times[0])/60 # in hours
        if total_time < 5 and total_time > 2:
            self.cost["coupon"] += 22.10
            self.coupon_histogram[0] += 1
        elif total_time < 10 and total_time > 5:
            self.cost["coupon"] += 28.20
            self.coupon_histogram[1] += 1
        elif total_time < 20 and total_time > 10:
            self.cost["coupon"] += 35.30
            self.coupon_histogram[2] += 1
        elif total_time > 20:
            self.cost["coupon"] += 35.30*2
            self.coupon_histogram[3] += 1


    def findClosestHub(self):
        """
        Function for the first policy.
        By the order's destination, calculate the euclidean distances between the destination and the
        different hubs, and find the closest hub.
        :return: The vector (point) of the closest hub to the destination.
        """
        hub_locations = [(500,2000),(3000,5500),(5500,2000)]
        list_of_hubs = [] # will be sorted later
        for i in range(len(hub_locations)):
            current_distance = 0

            for j in range(len(self.destination)): # Calculate the euclidean distance
                current_distance += math.pow(hub_locations[i][j] - self.destination[j], 2)

            list_of_hubs.append((i+1,math.sqrt(current_distance))) # add to the list a tuple: (num_of_hub,distance)

        # sort the data
        list_of_hubs = sorted(list_of_hubs, key=lambda x: x[1])
        # Set the data
        self.hub_location, self.distance_from_hub = (list_of_hubs[0][0]),list_of_hubs[0][1]

    def distancesFromHubs(self):
        """
        Function for the second policy.
        Sort the drone stations from the closest to the destionation to the farest.
        :return: List of tuples with the hubs' points, sorted from the closest to the destination to the farest.
        """
        hub_locations = [(500,2000),(3000,5500),(5500,2000)]
        list_of_hubs = [] # will be sorted later
        for i in range(len(hub_locations)):
            current_distance = 0

            for j in range(len(self.destination)): # Calculate the euclidean distance
                current_distance += math.pow(hub_locations[i][j] - self.destination[j], 2)

            list_of_hubs.append((i+1,math.sqrt(current_distance))) # add to the list a tuple: (num_of_hub,distance)

        # return the sorted list of the hubs - so the list will include tuples of the hubs, sorted from
        # the closest to the destination to the farest.
        return sorted(list_of_hubs, key=lambda x: x[1])

    def updateDroneType(self, dronetype):
        """
        Updates the drone type, given the drontype variable. Right after that, calculate the service time needed.
        :param dronetype: 1 - basic drone, 2 - regular drone, 3 - super drone
        :return:
        """
        self.drone_type = dronetype
        self.calculateServiceTime()

    def calculateServiceTime(self):
        """
        Calculates the relevant service time and the time needed to pass the one-way distance to the destination from
        the hub.
        :return:
        """
        if self.drone_type == 1:
            handling_time = np.random.uniform(10, 30)  # minutes
        elif self.drone_type == 2:
            handling_time = float(5 + np.random.exponential(5))  # minutes
        else:
            handling_time = float(10 + np.random.exponential(10))  # minutes

        # Euclidean time
        distance_time = (self.distance_from_hub / 500) # in minutes

        self.service_time = handling_time
        self.one_way_time = distance_time

    def relevantRepairTime(self):
        """
        Repair time according to the drone type.
        :return:
        """
        if self.drone_type == 1:
            return np.random.uniform(1*60,3*60)
        elif self.drone_type == 2:
            return np.random.triangular(2*60, 3*60, 6*60)
        else:
            return np.random.triangular(2*60, 4*60, 8*60)

    def isDeliveredInAnotherDay(self):
        """
        Calculate after the order times have been set, whether the order has been delivered in the same day or not.
        :return:
        """
        if np.ceil(self.order_times[0] / (24*60)) != np.ceil(self.order_times[1] / (24*60)):
            self.delivered_in_another_day = 1



###############################################
####### Main drone company
###############################################
class DroneCompany():
    """
    The main company which manages the four hubs we have in this simulation.
    """
    def __init__(self, env):
        self.env = env
        self.hubs = [DroneStation(env, (500,2000)), DroneStation(env, (3000,5500)), DroneStation(env, (5500,2000))]
        self.rate = (150/60) # Same rate for all drone hubs. The default rate is assigned for 05:00-10:00
        self.orderList = []    # a list of all orders the drone company has gotten.

    def changeRate(self, option):
        """
        This function applys a change to the rate of the coming orders.
        :param options: 1,2,3,4,5
        :return:
        """
        if option == 1:       # change to 1; means 150 customers per hour (150/60 per minute) - 05:00-10:00
            self.rate = 150/60
        elif option == 2:     # change to 2; means 200 customers per hour (200/60 per minute) - 10:00-16:00
            self.rate = 200/60
        elif option == 3:     # change to 3; means 80 customers per hour (80/60 per minute) - 16:00 - 22:00
            self.rate = 80/60
        elif option == 4:     # change to 4; means 0 customers per hour 22:00 - 05:00
            self.rate = 0
        elif option == 5:     # change to 5; means the week has finished, no more customers until the end of simulation.
            self.rate = -1

        yield self.env.timeout(0)

    def getRelevantDroneType(self, order, policy):
        """
        According to the relevant policy, decide what is the drone type that is going to deliver the order.
        :return: drone type: 1 - basic, 2 - regular, 3 - super.
        """
        ########################### FIRST POLICY #########################
        if policy == 1:
            order.findClosestHub()  # according to the policy, find the closest hub.
            hub = (order.hub_location)-1
            if order.size == 1:
                if self.hubs[hub].operation_centers[0].count < 20:
                    return 1
                elif self.hubs[hub].operation_centers[1].count < 7:
                    return 2
                else:
                    return 1
            elif order.size == 2:
                if self.hubs[hub].operation_centers[1].count < 7:
                    return 2
                elif self.hubs[hub].operation_centers[2].count < 3:
                    return 3
                else:
                    return 2
            else:
                return 3

#################################################################################################
######################################### FUNCTIONS #############################################
#################################################################################################

###############################################
####### Raffle Destination
###############################################
def raffleDestination():
    """
    Calculate the destination of a new order.
    :return: The point of the order's destination
    """
    raffle = np.random.uniform(0, 1) # choose between city center or city periphery
    if raffle > 0.5:
        point = (np.random.uniform(2000, 4000), np.random.uniform(2000, 4000)) # city center
    else:
        point = (np.random.uniform(0, 6000), np.random.uniform(0, 6000)) # city periphery
        while (point[0] <= 4000 and point[0] >= 2000) and (point[1] <= 4000 and point[1] >= 2000): # must be outside of the city center
            # center
            point = (np.random.uniform(0, 6000), np.random.uniform(0, 6000))
    return point


###############################################
####### Orders Arrival
###############################################
def ordersArrival(env, drone_company):
    """
    The main simulation's process. Generates the upcoming orders according to the drone company's rate.
    :param env: the simulation environment
    :param drone_company: the relevant drone company for this simulation
    :return:
    """

    while True: # as long as new orders should arrive:
        if env.now == 0: # have we just started the simulation?
            yield env.timeout(5*60) # wait until 05:00AM
            continue
        elif drone_company.rate == 0: # are we in the time period 22:00-05:00?
            # no more orders are coming untill morning.
            yield env.timeout(7*60) # Wait 7 hours, the relevant time until 05:00 AM
            continue
        elif drone_company.rate == -1: # our week has finished!
            break # stop the orders arrival.

        yield env.timeout(np.random.exponential(1/drone_company.rate)) # orders arrive at the drone company's rate.
        new_order = Order(env, raffleDestination()) # create a new order
        new_order.updateDroneType(drone_company.getRelevantDroneType(new_order, POLICY)) # get the drone type for
        # the current order according to the relevant POLICY
        new_order.order_times[0] = env.now # catch current time when the order has been received
        env.process(orderExecution(env, new_order, drone_company)) # process the order



###############################################
####### Order's Execution
###############################################
def orderExecution(env, order, drone_company):
    """
    The orders execution function. After the relevant data has been processed, begin the process of the order.
    :param env: Relevant simulation environment
    :param order: The order to process
    :param drone_company: The drone company of the simulation
    :return:
    """
    # process the order at his assigned hub and operation center (according to the policy)
    with drone_company.hubs[(order.hub_location)-1].operation_centers[order.drone_type-1].request() as my_turn:
        yield my_turn # wait, if needed, to your turn.
        yield env.timeout(order.one_way_time + order.service_time) # order is now processed!
        order.order_times[1] = env.now # catch current time when the order has been delivered
        order.delivered = 1 # mark as delivered

        ### COSTS ###
        order.updateServiceAndDistanceCost() # update service and distance costs

        ### DELIVERED IN THE SAME DAY? ###
        order.isDeliveredInAnotherDay() # check if the order was delivered in the same day it had been ordered

        # A problem has occurred that has to be fixed?
        problem = np.random.uniform(0,1)
        if problem < 0.1: # if so - the drone will return to service only after the repair is finished
            yield env.timeout(order.one_way_time + order.relevantRepairTime())
        else: # otherwise; just get back to the station already!
            yield env.timeout(order.one_way_time)
        order.coupon() # calculate the relevant coupon cost
        drone_company.orderList.append(order) # add the order to the drone company and release the resource.



#################################################################################################
#################################### MAIN SIMULATION ############################################
#################################################################################################

######### HISTOGRAM:
# 1 cell = 0-2 hours service
# 2 cell = 2-4 hours service
# 3 cell = 4-6 hours service
# 4 cell = 6-8 hours service
# 5 cell = 8-10 hours service
# 6 cell = longer than 10 hours
histogram = [0,0,0,0,0,0]

######### MEASUREMENTS:
total_cost_for_graph = []
orders_delayed_for_graph = []
coupon_histogram_for_graph = [0,0,0,0]
measurements = [0,0,0,0,0,0]
# 1 - Distance cost
# 2 - Service cost
# 3 - Coupon cost
# 4 - orders that took more than 2 hours to deliver


print("Policy %s:" % (POLICY))

#### MAIN LOOP
week = 1
while week <= 923:
    #### Set simulation environment
    env = simpy.Environment()
    drone_company = DroneCompany(env)

    ############################## Rate changes schedule ############################################
    for i in range(0,5):
        mult = i*24
        start_delayed(env, drone_company.changeRate(1), ((5+mult) * 60))
        start_delayed(env, drone_company.changeRate(2), (10+mult)*(60))
        start_delayed(env, drone_company.changeRate(3), (16+mult)*(60))
        if i != 4:
            start_delayed(env, drone_company.changeRate(4), (22+mult)*(60))
        else:
            start_delayed(env, drone_company.changeRate(5), (22+mult)*60)
    ############################## Rate changes schedule ############################################

    env.process(ordersArrival(env, drone_company))
    env.run(until=168*60) # the simulation will run until all orders have been processed.

    # calculate how much days, hours and minutes the simulation was running.
    days = np.floor(env.now/(60*24))
    hours = np.floor((env.now/60)-(days*24))
    minutes = ((env.now/60)-(days*24)-hours)*60
    print("\nWeek no. %s: Simulation duration is: %s days, %s hours, %s minutes" % (week,days,hours,minutes))
    #if days >= 7:
    #    continue # run the simulation again because we exceeded the time limit of 7 days

    # initiate counters
    orders_delayed_this_week = 0
    total_cost_this_week = 0

    for i in range(len(drone_company.orderList)):  # for each order
        order_time = (drone_company.orderList[i].order_times[1] - drone_company.orderList[i].order_times[0]) / 60
        if drone_company.orderList[i].delivered_in_another_day:
            orders_delayed_this_week += 1 # increase the counter of delayed deliveries

        measurements[0] += drone_company.orderList[i].cost["distance"] # catch the distance cost of this order
        measurements[1] += drone_company.orderList[i].cost["service"] # catch the service cost of this order
        measurements[2] += drone_company.orderList[i].cost["coupon"] # catch the coupon cost of this order
        total_cost_this_order = drone_company.orderList[i].cost["distance"] + drone_company.orderList[i].cost["service"] + drone_company.orderList[i].cost["coupon"]
        total_cost_this_week += total_cost_this_order # add this order's cost to the total week cost

        if order_time > 2:
            measurements[3] += 1 # catch the orders that took more than 2 hours to deliver

        # building the histogram: for the current order, increase the counter for the relevant histogram slot.
        if order_time < 2: # 0-2 hours period
            histogram[0] += 1
        elif order_time < 4: # 2-4 hours period
            histogram[1] += 1
        elif order_time < 6: # 4-6 hours period
            histogram[2] += 1
        elif order_time < 8: # 6-8 hours period
            histogram[3] += 1
        elif order_time < 10: # 8-10 hours period
            histogram[4] += 1
        elif order_time > 10: # More than 10 hours period
            histogram[5] += 1

        # bulilding the histogram for coupon cost
        for j in range(len(coupon_histogram_for_graph)):
            coupon_histogram_for_graph[j] += drone_company.orderList[i].coupon_histogram[j]

    ### For visualization: add the current week's total cost and delayed orders to a list in their right order
    ### for doing that, we use insert to the last location, so each place in the list has the relevant data
    ### for the current week: 0 - first week, 1 - second week and so on until 99 - the 100th week.
    total_cost_for_graph.insert(len(total_cost_for_graph), total_cost_this_week)
    orders_delayed_for_graph.insert(len(orders_delayed_for_graph), orders_delayed_this_week)

    week += 1 # increasing the week counter in order to continue the loop for the next week

### Simulation output:
print("Orders that did not deliver in the same day: %s\nMean of total cost: %s" % (
sum(orders_delayed_for_graph) / len(orders_delayed_for_graph), sum(total_cost_for_graph) / len(total_cost_for_graph)))


############################################################
######### HISTOGRAM GRAPH
############################################################
x_axis = ["0 to 2\nhours", "2 to 4\nhours", "4 to 6\nhours", "6 to 8\nhours", "8 to 10\nhours", "more than 10\nhours"]
plt.bar(range(len(histogram)), histogram, align='center', color='b')
plt.xticks(range(len(histogram)), x_axis)
plt.xlabel('Periods')
plt.ylabel('Amount of Orders')
plt.title('Histogram for delivery times - policy ' + str(POLICY))
plt.show()
print("histogram list: %s\nTotal orders: %s" % (histogram,sum(histogram)))

############################################################
######### COUPON TYPE HISTOGRAM GRAPH
############################################################
x_axis = ["2 to 5\nhours", "5 to 10\nhours", "10 to 20\nhours", "more than 20\nhours"]
plt.bar(range(len(coupon_histogram_for_graph)), coupon_histogram_for_graph, align='center', color='b')
plt.xticks(range(len(coupon_histogram_for_graph)), x_axis)
plt.xlabel('Periods')
plt.ylabel('Amount of Orders')
plt.title('Histogram for coupon payment - policy ' + str(POLICY))
plt.show()

### Visualization:
############################################################
######### TOTAL COST PLOT GRAPH
############################################################
x_axis = [i for i in range(len(total_cost_for_graph))]
plt.plot(x_axis, total_cost_for_graph, color='b')
plt.xlabel('Weeks')
plt.ylabel('Amount in ₪')
plt.title('Service, coupon & distance costs - policy ' + str(POLICY))
plt.show()

############################################################
######### Orders that did not deliver in the same day GRAPH
############################################################
x_axis = [i for i in range(len(orders_delayed_for_graph))]
plt.plot(x_axis, orders_delayed_for_graph, color='b')
plt.xlabel('Weeks')
plt.ylabel('Amount of Orders')
plt.title('Orders that did not deliver in the same day - policy ' + str(POLICY))
plt.show()