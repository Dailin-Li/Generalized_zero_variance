#%%
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
#%%
#convergence of probabiltiy distribution
transition_matrix = np.array([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]])
initial_vector = np.array([[0,0,1]])

state1 = []
state2 = []
state3 = []
for i in range(50): 
    state1.append(initial_vector[0][0])
    state2.append(initial_vector[0][1])
    state3.append(initial_vector[0][2])
    initial_vector = np.dot(initial_vector,transition_matrix)

x = np.arange(50)
plt.plot(x,state1,label='State 1') 
plt.plot(x,state2,label='State 2')
plt.plot(x,state3,label='State 3')
plt.legend(loc = "upper right")
plt.xlabel("Iterations")
plt.ylabel("Probabiltiy of states")
plt.show()

#%%
#convergence of n-step transition matrix
state1 = []
state2 = []
state3 = []
state4 = []
state5 = []
state6 = []
state7 = []
state8 = []
state9 = []
transition_matrix = np.array([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]])
matrix = np.array([[0.9,0.075,0.025],[0.15,0.8,0.05],[0.25,0.25,0.5]])
x = np.arange(100)
for i in range(100):
    
    state1.append(matrix[0][0])
    state2.append(matrix[0][1])
    state3.append(matrix[0][2])
    state4.append(matrix[1][0])
    state5.append(matrix[1][1])
    state6.append(matrix[1][2])
    state7.append(matrix[2][0])
    state8.append(matrix[2][1])
    state9.append(matrix[2][2])
    matrix = np.dot(matrix,transition_matrix)

plt.plot(x,state1,label='P_11')
plt.plot(x,state2,label='P_12')
plt.plot(x,state3,label='P_13')
plt.plot(x,state4,label='P_21')
plt.plot(x,state5,label='P_22')
plt.plot(x,state6,label='P_23')
plt.plot(x,state7,label='P_31')
plt.plot(x,state8,label='P_32')
plt.plot(x,state9,label='P_33')
plt.legend(loc = "upper right")
plt.xlabel("Iterations")
plt.ylabel("Matrix state")
plt.show()

#%%
#walk example

fig_size=(8,8)
figsize=fig_size

iterations=20
x=list(range(iterations+1))
init=-10
#plot 1:
error= np.random.randn(iterations)
sequence=[init]
for i in range(iterations):
    sequence+=[sequence[i]/2+error[i]]

plt.subplot(2, 2, 1)
plt.ylim(ymin = -10)
plt.ylim(ymax = 10)  
plt.ylabel("State value")  
plt.scatter(x,sequence)

#plot 2:
error= np.random.randn(iterations)
sequence=[init]
for i in range(iterations):
    sequence+=[sequence[i]/2+error[i]]

plt.subplot(2, 2, 2)
plt.ylim(ymin = -10)
plt.ylim(ymax = 10)    
plt.scatter(x,sequence)

init=10
#plot 3:
error= np.random.randn(iterations)
sequence=[init]
for i in range(iterations):
    sequence+=[sequence[i]/2+error[i]]

plt.subplot(2, 2, 3)
plt.ylim(ymin = -10)
plt.ylim(ymax = 10)    
plt.ylabel("State value")  
plt.scatter(x,sequence)
plt.xlabel("Iterations")  
#plot 4:
error= np.random.randn(iterations)
sequence=[init]
for i in range(iterations):
    sequence+=[sequence[i]/2+error[i]]

plt.subplot(2, 2, 4)
plt.ylim(ymin = -10)
plt.ylim(ymax = 10)   
plt.xlabel("Iterations")   
plt.scatter(x,sequence)

plt.show()

#%%
#perpare to plot the histogram

#define the target distribution function
def pi(x):
    return (norm.pdf(x, loc=-5, scale=3)+
            norm.pdf(x, loc=5, scale=3))/2


N = 200000  # Number of iterations
sample = [0 for i in range(N)] # set the chain
sample[0] = 0  # set the Initial start point

# Loops for random walk along the chain based on Q(y|x)
for t in range(1, N):
    x = sample[t - 1]
    y = norm.rvs(loc=x, scale=1, size=1)[0] #Sample Next point 
    
    #Metropolis-Hastings acceptance probability
    alpha = min(1, (pi(y) / pi(x)))  
    
    u = random.uniform(0, 1)  # sample u following U(0,1)
    # Accept/Reject
    if u < alpha:  
        sample[t] = y
    else:
        sample[t] = x
       
mean = np.mean(sample)
median = np.median(sample)
perc_25, perc_75 = np.percentile(sample, 25), np.percentile(sample, 75)
  
#%% histgram

x = np.arange(-15, 15, 0.01)
#plot the target distribution
plt.plot(x, pi(x), color='r',label='Target Distribution') 
#plot the histogram of sampling
plt.hist(sample, bins=100,density=True, color='b', 
         edgecolor='k', alpha=0.6,label='Samples Distribution')  
plt.legend()
#plt.title("Histgram of Example Using Metropolis-Hasting Sampling",weight="bold")
plt.axvline(mean, color='r', lw=2, linestyle='--')
plt.axvline(perc_25, linestyle=':', color='k', alpha=0.5,linewidth=3)
plt.axvline(perc_75, linestyle=':', color='k', alpha=0.5,linewidth=3)
plt.xlabel("Sampled value")
plt.ylabel("Value")
plt.show()
#%% trace plot

plt.plot(sample)
plt.xlabel('Samples')
plt.ylabel('Variable')
plt.axhline(mean, color='r', lw=2, linestyle='--')
#plt.axhline(median, color='c', lw=2, linestyle='--')
plt.axhline(perc_25, linestyle=':', color='k', alpha=0.5,lw=1,linewidth=3)
plt.axhline(perc_75, linestyle=':', color='k', alpha=0.5,lw=1,linewidth=3)
#plt.title("Trace Plot of Example Using Metropolis-Hasting Sampling",weight="bold")

