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
    int label[n];
    real length[n];
    real widthLeft[n];
    real rightEdge[n];
    real bottomMargin[n];
}
transformed data {}
parameters {
    real coef1;
    real coef2; 
    real coef3;
    real coef4;
    real coef5;

}
transformed parameters {}
model {
    coef1 ~ normal(0, 100);
    coef2 ~ normal(0, 100); 
    coef3 ~ normal(0, 100);
    coef4 ~ normal(0, 100);
    coef5 ~ normal(0, 100);

    for(i in 1:n) {
        label[i] ~ bernoulli(Phi(coef1*length[i] + coef2*widthLeft[i] + coef3*rightEdge[i] + coef4*bottomMargin[i] + coef5));
  }
}
generated quantities {}
"""
 

model = pystan.StanModel(model_code=log_reg_code)

df = pd.read_csv('data/swiss.csv')
df.head()

log_reg_dat = {
             'n': len(df),
             'label': df.label,
             'length': df.length,
             'widthLeft': df.widthLeft,
             'rightEdge': df.rightEdge,
             'bottomMargin': df.bottomMargin,
            }

result = model.sampling(data=log_reg_dat, iter=55000,warmup=5000 ,chains=1)

end = time.time()
MCMC_sample_time=end-start

# take parameters
take_para = result.extract()
coef1 = take_para['coef1']
coef2 = take_para['coef2']
coef3 = take_para['coef3']
coef4 = take_para['coef4']
coef5 = take_para['coef5']

# obtain mcmc samples
mcmc_samples = []
for i in range(len(coef1)):
    mcmc_samples.append([coef1[i], coef2[i], coef3[i], coef4[i],coef5[i]]) 
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

