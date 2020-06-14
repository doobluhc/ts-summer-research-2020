#!/usr/bin/env python
# coding: utf-8

# In[6]:


from control.matlab import *
import numpy as np
import slycot
import matplotlib.pyplot as plt


# In[73]:


#this method returns the average cost trajectory, predicted a, predicted b and the ratio of a and b
def adaptive_algorithm(T,a,b):
    C_t = [0] * T # list of average cost
    c_t = 0
    x_t = [0] * (T+1) #list of states
    a_hat_b_hat = [0] * T #list of ratio of a and b 
    a_hat = [0] * T # list of predicted a
    b_hat = [0] * T# list of predicted b 
    theta_t = [np.array([[a],[b]])] * (T+1) #list of  predicted theta
    u_t = [0] * T #list of predicted control input
    sigma_t = [0] * (T+1)
    sigma_t[0] = 1
    for t in range(T):
        #update control input
        u_t[t] = -theta_t[t][0][0]/theta_t[t][1][0] * x_t[t] 
        #disturbance
        w_t = np.random.normal(0.0,1.0) 
        #update state
        x_t[t+1] = a * x_t[t] + b * u_t[t] + w_t 
        #update cost
        c_t = c_t + x_t[t]**2 
        C_t[t] = c_t/(t+1)
        a_hat_b_hat[t] = theta_t[t][0][0]/theta_t[t][1][0]
        a_hat[t] = theta_t[t][0][0]
        b_hat[t] = theta_t[t][1][0]
        z_t_transpose = np.array([[x_t[t],u_t[t]]])
        z_t = np.array([[x_t[t]],[u_t[t]]])
        theta_t[t+1] = theta_t[t] + sigma_t[t]*z_t*(x_t[t+1] - np.dot(z_t_transpose,theta_t[t]))/(1+np.dot(np.dot(z_t_transpose,sigma_t[t]),z_t))
        sigma_t[t+1] = sigma_t[t] - sigma_t[t]*np.dot(z_t,z_t_transpose)*sigma_t[t]/(1+z_t_transpose*sigma_t[t]*z_t)
    return C_t,a_hat_b_hat,a_hat,b_hat


# In[74]:


a = 0.3 #starting value for a
b = 0.1 #starting value for b
T = 1000 #number of run times
N = 100 #number of trials 
cost = [[0] * T for i in range(N)]
a_hat_b = [[0] * T for i in range(N)]
a_hat = [[0] * T for i in range(N)]
b_hat = [[0] * T for i in range(N)]
for n in range(N):
    cost[n],a_hat_b[n],a_hat[n],b_hat[n]= adaptive_algorithm(T,a,b)


# In[75]:


#plot the data
plt.title("cost function value/t vs t (a = 3, b =1)")
plt.xlabel('t')
plt.ylabel('cost/t')
for n in range(N):
    plt.plot([t for t in range(T)],cost[n])


# In[78]:


#plot the data
plt.title("hat a_t/\hat b_t vs t (a = 3, b =1)")
plt.xlabel('t')
plt.ylabel('hat a_t/\hat b_t')
for n in range(N):
    plt.plot([t for t in range(T)],a_hat_b[n])
    plt.axis([0,T,0,100])


# In[79]:


#plot the data
plt.title("hat a_t vs t (a = 3, b =1)")
plt.xlabel('t')
plt.ylabel('hat a_t')
for n in range(N):
    plt.plot([t for t in range(T)],a_hat[n])


# In[80]:


#plot the data
plt.title("hat b_t vs t (a = 3, b =1)")
plt.xlabel('t')
plt.ylabel('hat b_t')
for n in range(N):
    plt.plot([t for t in range(T)],b_hat[n])

