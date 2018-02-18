import csv
import numpy as np
import matplotlib.pyplot as plt
training = []        #training data
data = []           #converting the csv file to an array to better handle data
testing = []        #data for testing
'''
Get open and close prices from the past 10 years of daily Apple stock data, reverse them so that the array would be in chronological order
'''
with open('HistoricalQuotes.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    next(csvreader)
    for row in csvreader:
        data.append(row)
    counter = 0
    for x in reversed(data):
        if counter <  2* len(data)/3:       #set 66% of data for training and 33% for testing
            training.append(x)
        else:
            testing.append(x)
        counter+=1

class Queue():              #data structure for reward matrix
    def __init__(self,length):
        self.max_length = length
        self.data = [0]
    def add(self,amt):
        if(len(self.data) == self.max_length):          #ensures length of data kept at max
            self.data.pop(0)                            #first in first out
        self.data.append(amt)
    def getLast(self):
        return self.data[len(self.data)-1]
    def getMean(self):
        return np.mean(self.data)
'''
Initializing reward and q learning matrices: None indicates we can't perform the action, such as buying when we've already bought our stock, or selling when we've already sold it
Setup:                  
                Long   [Stay on current state       Sell                      ]
                Short  [Buy                         Stay on current state     ]
                Neutral[Buy                         Sell                      ]
Long - Buying a stock and then later selling it at a higher price
Short- Selling a stock and then later buying it a lower price
Neutral - when neither of these actions are performed
'''
class Trader():

    def __init__(self):
        self.mean_window = 50           #window of time for reward matrix
        self.net_returns = 0            #total returns from trading
        self.dr = .5                    #discount rate(how much Q-matrix takes into account future actions)
        self.epochs = 30                #number of training cycles
        self.reward_matrix = [[Queue(self.mean_window),Queue(self.mean_window)],        #Queue for each state-action to store averages
                              [Queue(self.mean_window),Queue(self.mean_window)],
                              [Queue(self.mean_window),Queue(self.mean_window)]]
        self.q_matrix =      [[0,0],
                              [0,0],
                              [0,0]]

    def choose_action(self,state):
        if np.abs(self.q_matrix[state][0] - self.q_matrix[state][1]) < .00001:            #if values are the same
            return np.random.randint(0,2)                                                 #random either 0 or 1
        else:
            if self.q_matrix[state][0] > self.q_matrix[state][1]:
                return 0
            else:
                return 1

    def update_reward_matrix(self,state,action,prev_price,cur_price):
        if state != -1 and action != -1:            #previous state and action exist
            if action == 0:
                change = cur_price - prev_price     #price desired to go up after buying
            elif action == 1:
                change = prev_price - cur_price     #price desired to go down after selling
            self.reward_matrix[state][action].add(change)

    def update_q_matrix(self,state,action):
        if state != -1 and action != -1:            #previous state and action exist
            self.q_matrix[state][action] = self.reward_matrix[state][action].getMean() + self.dr * self.q_matrix[action][self.choose_action(action)]

    def update_my_returns(self,amt,action):
        if action == 0:
            self.net_returns -= amt         #buying stock = giving money
        else:
            self.net_returns += amt         #selling stock = gaining money

    def reset(self):                        #reset Q and reward matrices
        self.net_returns = 0
        self.q_matrix = np.zeros(np.shape(self.q_matrix))
        self.reward_matrix = [[Queue(self.mean_window),Queue(self.mean_window)],
                              [Queue(self.mean_window),Queue(self.mean_window)],
                              [Queue(self.mean_window),Queue(self.mean_window)]]

    def train(self,train_data):
        mean_q_matrix = []
        max_val = 0
        max_q = []
        for epochs in range(self.epochs):
            cur_state = 2  # start at neutral
            prev_state = -1  # previous state(doesn't exist yet)
            prev_action = -1  # previous action(doesn't exist yet)
            cur_price = 0.0  # current open price on given day
            prev_price = 0.0  # yesterday's price
            order_final = 0  # price at time when share is either bought or sold to end loop

            for day in train_data:
                cur_price = float(day[3])  # current open price on given day
                self.update_reward_matrix(prev_state, prev_action, prev_price, cur_price)
                self.update_q_matrix(prev_state, prev_action)
                cur_action = self.choose_action(cur_state)  # current action performed
                prev_state = cur_state  # previous state
                prev_action = cur_action  #previous action
                if cur_state == cur_action: #buy/sell loop is maintained
                    cur_state = prev_state
                elif prev_state == 2:       #in neutral
                    order_initial = cur_price  # initial open price that share was bought or sold to start loop
                    self.update_my_returns(order_initial, cur_action)
                    cur_state = cur_action
                else:                          #buy/sell cycle ends
                    cur_state = 2              #revert back to neutral
                    order_final = cur_price  # end open pice that share was bought or sold to end loop
                    self.update_my_returns(order_final, cur_action)
                prev_price = cur_price

            mean_q_matrix.append([self.q_matrix, self.net_returns])
            self.reset()
        for x in mean_q_matrix:                #get "max Q" based on maximum returns
            if x[1] > max_val:
                max_val = x[1]
                max_q = x[0]
        self.q_matrix = max_q
        self.net_returns = 0

    def test(self,test_data):
        cur_state = 2  # start at neutral
        prev_state = -1  # previous state(doesn't exist yet)
        prev_action = -1  # previous action(doesn't exist yet)
        cur_price = 0.0  # current open price on given day
        prev_price = 0.0  # yesterday's price
        order_final = 0  # price at time when share is either bought or sold to end loop
        plot_returns = []
        for day in test_data:
            cur_price = float(day[3])  # current open price on given day
            cur_action = self.choose_action(cur_state)  # current action performed
            prev_state = cur_state  # previous state
            prev_action = cur_action
            if cur_state == cur_action:
                cur_state = prev_state
            elif prev_state == 2:
                order_initial = cur_price  # initial open price that share was bought or sold to start loop
                self.update_my_returns(order_initial, cur_action)
                print(self.net_returns,order_initial,cur_action)
                cur_state = cur_action
            else:
                cur_state = 2
                order_final = cur_price  # end open pice that share was bought or sold to end loop
                self.update_my_returns(order_final, cur_action)
                print(self.net_returns,order_final,cur_action)
            prev_price = cur_price
            plot_returns.append([self.net_returns])
        plt.plot(plot_returns)
        plt.show()
        for row in self.q_matrix:
            print row
        print self.net_returns

experiment_1 = Trader()
experiment_1.train(training)
experiment_1.test(testing)

