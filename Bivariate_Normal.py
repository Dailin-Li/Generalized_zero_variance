import pystan
import numpy as np
from Zero_variance_variation.zv_var import construct_A_mat, construct_cv_sample, cv_vd
import time
#%%
start = time.time() #start time for sampling
norm_code = """
data {
    int<lower=0> n;
    vector[2] y[n];
}
transformed data {}
parameters {
    vector[2] mu;
    real sig_11;
    real sig_22;
    real sig_12;
}
transformed parameters {
    matrix[2, 2] sig;
    sig[1, 1] = sig_11;
    sig[1, 2] = sig_12;
    sig[2, 1] = sig_12;
    sig[2, 2] = sig_22;
}
model {
    y ~ multi_normal(mu, sig);
}
generated quantities {}
"""
model = pystan.StanModel(model_code=norm_code)

norm_dat = {
             'n': 1000,
             'y': np.random.multivariate_normal(np.asarray([10, 20]), np.array([[4,2],[2,6]]), 1000),
            }

result = model.sampling(data=norm_dat,warmup=5000,chains=1, iter=55000, verbose=False)

end = time.time() #end time for sampling
MCMC_sample_time=end-start #store time for sampling

# take parameters
take_para = result.extract()
mu = take_para['mu']
sig_11 = take_para['sig_11']
sig_22 = take_para['sig_22']
sig_12 = take_para['sig_12']

# obtain mcmc samples
mcmc_samples = []
for i in range(len(mu)):
    mu_list = mu[i].tolist()
    mu_list.extend([sig_11[i], sig_22[i], sig_12[i]])
    mcmc_samples.append(mu_list)
mcmc_samples = np.asarray(mcmc_samples)

# obtain gradients of log-posterior
grad_log_prob_val = []
for i in range(len(mu)):
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
#%%
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
    labels =['Orginal','mode1','mode2','mode3','mode4','mode5','mode6','mode7','mode8']
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
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

for i in range(1):
    cv_sample_list=[]
    for j in range(3):
        cv_sample_value=cv_sample_dict['cv_sample_'+str(j+1)]
                             
        cv_sample_list+=[cv_sample_value[:,i]]
    
#all_data = [np.random.normal(0, std, size=100) for std in range(1, 4)]
labels = ['x1', 'x2', 'x3']
#%%
fig, ax = plt.subplots(1, figsize=fig_size)
bplot = ax.boxplot( cv_sample_list, patch_artist=True, labels=labels,showfliers=False)  # 设置箱型图可填充
plt.title('Rectangular box plot')

colors = ['pink', 'lightblue', 'lightgreen']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)  

plt.grid(True)
plt.xlabel('Three separate samples')
plt.ylabel('Observed values')
plt.show()


