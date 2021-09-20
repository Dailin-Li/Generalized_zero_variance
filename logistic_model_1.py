import pystan
import numpy as np
import pandas as pd
from Zero_variance_variation.zv_var import construct_A_mat, construct_cv_sample, cv_vd
import time
#%%
start = time.time()

log_reg_code = """
data {
    int<lower=0> n;
    int male[n];
    real weight[n];
    real height[n];
}
transformed data {} 
parameters {
    real coef1;
    real coef2;
    real coef3;
}
transformed parameters {}
model {
    coef1 ~ normal(0, 100);
    coef2 ~ normal(0, 100);
    coef3 ~ normal(0, 100);
    for(i in 1:n) {
        male[i] ~ bernoulli(inv_logit(coef1*weight[i] + coef2*height[i] + coef3));
  }
}
generated quantities {}
"""
model = pystan.StanModel(model_code=log_reg_code)

df = pd.read_csv('data/HtWt.csv')
df.head()

log_reg_dat = {
             'n': len(df),
             'male': df.male,
             'height': df.height,
             'weight': df.weight
            }

result = model.sampling(data=log_reg_dat, iter=55000,warmup=5000, chains=1)

end = time.time()
MCMC_sample_time=end-start

# take parameters
take_para = result.extract()
coef1 = take_para['coef1']
coef2 = take_para['coef2']
coef3 = take_para['coef3']

# obtain mcmc samples
mcmc_samples = []
for i in range(len(coef1)):
    mcmc_samples.append([coef1[i], coef2[i], coef3[i]])
mcmc_samples = np.asarray(mcmc_samples)

# obtain gradients of log-posterior
grad_log_prob_val = []
for i in range(len(coef1)):
    grad_log_prob_val.append(result.grad_log_prob(mcmc_samples[i], adjust_transform=False))
grad_log_prob_val = np.asarray(grad_log_prob_val)
#%%
#construct "matrix C"
start = time.time() #start time for constructing matrix
A_mat=construct_A_mat(mcmc_samples,grad_log_prob_val)
end = time.time() #end time for constructing matrix
cv_run_time_matrix=end-start #store time for constructing matrix

#%%
#Three purposes:
cv_sample_dict=dict() #store the control variates (CV) samples
cv_run_time=np.zeros((8)) #calculate running time
cv_num=np.zeros((8)) #store number of coefficients of different modes

for i in range(8):
    
    start = time.time()
    cv_sample,num=construct_cv_sample(A_mat,mcmc_samples,mode=i+1) #optimize the coefficent
    end = time.time()
    
    cv_sample_dict['cv_sample_'+str(i+1)]=cv_sample   
    
    cv_num[i]=num    
    cv_run_time[i]=end-start

#store in "Standard form"
cv_run_time = cv_run_time_matrix*(cv_num/cv_num[7])+cv_run_time  
cv_run_time_store=[]
for i in range(cv_run_time.shape[0]):
    cv_run_time_store+=[str('%.3e' %cv_run_time[i])]   
#%%
#check the variance reduction ratio

cv_vd_dict=dict()
cv_vd_dict_store=dict()
for i in range(8): 

    cv_vd_value=cv_vd(cv_sample_dict['cv_sample_'+str(i+1)] ,mcmc_samples)  
    
    #store the variance reduction ratio
    cv_vd_dict['cv_vd_'+str(i+1)]=cv_vd_value
    cv_vd_store=[]
    
    #store in "Standard form"
    for j in range(cv_vd_value.shape[0]):
        cv_vd_store+=[str('%.3e' %cv_vd_value[j])]         
              
    cv_vd_dict_store['cv_vd_'+str(i+1)]=cv_vd_store

#%%
#check the measuring efficiency
cv_comput_cost=cv_run_time+MCMC_sample_time
cv_comput_cost_ratio=MCMC_sample_time/cv_comput_cost

cv_me_dict=dict()
cv_me_dict_store=dict()
for i in range(8): 

    cv_vd_value=cv_vd_dict['cv_vd_'+str(i+1)]
    cv_me_value=cv_comput_cost_ratio[i]/cv_vd_value
    cv_me_dict['cv_me_'+str(i+1)]=cv_me_value
    cv_me_store=[]
    
    #store in "Standard form"
    for j in range(cv_me_value.shape[0]):
        cv_me_store+=[str('%.3e' %cv_me_value[j])]         
              
    cv_me_dict_store['cv_me_'+str(i+1)]=cv_me_store

#%% 
#boxplot

import matplotlib.pylab as plt
fig_size=(8, 10)
num_samples = mcmc_samples.shape[0]
dim = mcmc_samples.shape[1]

fig, ax = plt.subplots(dim, figsize=fig_size)
for i in range(dim):
    
    cv_sample_list=[mcmc_samples[:,i]]
    for j in range(8):
        cv_sample_value=cv_sample_dict['cv_sample_'+str(j+1)]                             
        cv_sample_list+=[cv_sample_value[:,i]]
    labels =['Orginal','Type1','Type2','Type3','Type4','Type5','Type6','Type7','Type8']
    ax[i].boxplot( cv_sample_list, patch_artist=True, labels=labels,showfliers=False)
    ax[i].grid(True)

    ax[i].set_title('Parameter ' + str(i+1), loc='left')
fig.subplots_adjust(hspace=1.0)
#%% 
#traceplot

import matplotlib.pylab as plt
fig_size=(8, 8)
num_samples = mcmc_samples.shape[0]
dim = mcmc_samples.shape[1]

fig, ax = plt.subplots(dim, figsize=fig_size)
for i in range(dim):
    #plot the samples and different modes flexiblely
    ax[i].plot(range(0, num_samples), mcmc_samples[:,i], color='blue', 
                  label='Orginal Sample')
    ax[i].plot(range(0, num_samples), cv_sample_1[:,i], color='black',
                  label='Mode 1')
    ax[i].plot(range(0, num_samples), cv_sample_7[:,i], color='red',
                  label='Mode 7')
    ax[i].legend(loc=1, fancybox=True, bbox_to_anchor=(1.0, 1.5))
    ax[i].set_xlabel('Iterations')
    ax[i].set_ylabel('Parameter')
    ax[i].set_title('Parameter ' + str(i+1), loc='left')
fig.subplots_adjust(hspace=1.0)
#%%


#%%
#construct "matrix C"
start = time.time()
A_mat=construct_A_mat(mcmc_samples,grad_log_prob_val)
end = time.time()
spend_time[1]=end-start

#%%
#mode 1
start = time.time()
cv_sample_1,num_1=construct_cv_sample(A_mat,mcmc_samples,mode=1)
end = time.time()
spend_time[2]=end-start

#%%
cv_sample=dict()

#%%
#mode 2
start = time.time()
cv_sample_2,num_2=construct_cv_sample(A_mat,mcmc_samples,mode=2)
end = time.time()
spend_time[3]=end-start

#mode 3
start = time.time()
cv_sample_3,num_3=construct_cv_sample(A_mat,mcmc_samples,mode=3)
end = time.time()
spend_time[4]=end-start

#mode 4
start = time.time()
cv_sample_4,num_4=construct_cv_sample(A_mat,mcmc_samples,mode=4)
end = time.time()
spend_time[5]=end-start

#mode 5
start = time.time()
cv_sample_5,num_5=construct_cv_sample(A_mat,mcmc_samples,mode=5)
end = time.time()
spend_time[6]=end-start

#mode 6
start = time.time()
cv_sample_6,num_6=construct_cv_sample(A_mat,mcmc_samples,mode=6)
end = time.time()
spend_time[7]=end-start

#mode 7
start7 = time.time()
cv_sample_7,num_6=construct_cv_sample(A_mat,mcmc_samples,mode=7)
end = time.time()
spend_time[8]=end-start

#mode 8
start8 = time.time()
cv_sample_8,num_8=construct_cv_sample(A_mat,mcmc_samples,mode=8)
end = time.time()
spend_time[9]=end-start
#%%
def cv_vd(cv_sample,mcmc_samples):
    
    dim=mcmc_samples.shape[1] #dimension of parameters
    cv_vd_per=np.zeros((dim))         #(new sample variance)/(old sample variance) ratio
    for i in range(dim):        
        cv_vd_per[i]=np.var(cv_sample[:,i])/np.var(mcmc_samples[:,i])
    return cv_vd_per
#%%
cv_vd_1=cv_vd(cv_sample_1,mcmc_samples)
cv_vd_2=cv_vd(cv_sample_2,mcmc_samples)
cv_vd_3=cv_vd(cv_sample_3,mcmc_samples)
cv_vd_4=cv_vd(cv_sample_4,mcmc_samples)
cv_vd_5=cv_vd(cv_sample_5,mcmc_samples)
cv_vd_6=cv_vd(cv_sample_6,mcmc_samples)
cv_vd_7=cv_vd(cv_sample_7,mcmc_samples)
cv_vd_8=cv_vd(cv_sample_8,mcmc_samples)
#%% traceplot
import matplotlib.pylab as plt
fig_size=(8, 8)
num_samples = mcmc_samples.shape[0]
dim = mcmc_samples.shape[1]

fig, ax = plt.subplots(dim, figsize=fig_size)
for i in range(dim):
    ax[i].plot(range(0, num_samples), mcmc_samples[:,i], color='blue',
                  label='orginal sample')
    ax[i].plot(range(0, num_samples), cv_sample_1[:,i], color='black',
                  label='cv_sample_mode_1')
    ax[i].plot(range(0, num_samples), cv_sample_2[:,i], color='green',
                  label='cv_sample_mode_2')
    #ax[i].plot(range(0, num_samples), cv_sample_3[:,i], color='red',
     #             label='cv_sample_mode_3')
    ax[i].legend(loc=1, fancybox=True, bbox_to_anchor=(1.0, 1.5))
    ax[i].set_xlabel('mcmc iteration')
    ax[i].set_ylabel('Parameter')
    ax[i].set_title('Parameter ' + str(i+1), loc='left')
fig.subplots_adjust(hspace=1.0)
#%%