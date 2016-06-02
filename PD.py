import gym
import time
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
#from matplot import pyplot as plt


env = gym.make('Sphere-v0')

class PD(object):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev_err = 0
        self.prev_time = 0
        self.time_axis = []
        self.y_axis = []
    
    def ploting_data(self, time, x):
        self.time_axis.append(time)
        self.y_axis.append(x)
        
    def fx(self, error, x):
        self.cur_time = self.prev_time + 1
        dt = self.cur_time - self.prev_time
        de = error - self.prev_err
        
        self.cd = 0
        if dt > 0:
            self.cd = de/dt
        
        self.prev_time = self.cur_time
        self.prev_err = error
        res = self.kp*error + self.kd*self.cd
        self.ploting_data(self.cur_time, x)
        return res

def normalize(val):
    return ((val-min_val)/(float(max_val-min_val)))

pd_agent_x = PD(0.5, 10)
pd_agent_y = PD(0.5, 10)
x_goal, y_goal = (2, 2)
min_val, max_val = (-3, 3)
pos = env.reset()
action_x = 0.0
action_y = 0.0
x = []
y = []
#x = np.empty(shape=(1,1))
#y = np.empty(shape=(1,1))
#action = np.zeros(2)

for _ in range(500):
    #print("**********************")
    env.render()
    if(abs(pos[0] - x_goal) < 0.05 and abs(pos[1] - y_goal) < 0.05):
        x_goal = -2
        y_goal = -3
    
    error_x = x_goal - pos[0]    
    action_x = pd_agent_x.fx(error_x, pos[0])
    
    if(action_x > 3):
        action_x = 3
    elif (action_x < -3):
        action_x = -3
    error_y = y_goal - pos[1]    
    action_y = pd_agent_y.fx(error_y, pos[1])
    
    if(action_y > 3):
        action_y = 2.5
    elif (action_y < -3):
        action_y = -2.5
    
    action = np.array([action_x, action_y])
    pos = env.step(action)[0]

time = np.array(pd_agent_x.time_axis)
x = np.array(pd_agent_x.y_axis)
y = np.array(pd_agent_y.y_axis)
plt.plot(time, x)
plt.plot(time, y)
plt.annotate("kp=1, kd=5",
             xy=(50, 1.7),
             textcoords='offset points')
pl.show()
#print("+++++++++++++++++++++++++++++++++++")
#print(x)