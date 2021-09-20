import numpy as np
#from BasicFunction.Gaussian_log_prob import grad_log_prob


def construct_A_mat(samples,der_samples):
    """
    @samples: mcmc(hmc) samples
    @der_samples: log-graident of samples
    """    
    dim_var=samples.shape[1] #dimension of parameters
    num_var=samples.shape[0] #number of samples
    
    A_mat=np.zeros((dim_var,dim_var+1,num_var))
    
    #construct matrix "C' in the paper (including multiple samples)
    for i in range(dim_var):
        for j in range(dim_var+1):
            if j==0:
                A_mat[i,j,:]=der_samples[:,i]
            else:
                if i==j-1:
                    A_mat[i,j,:]=der_samples[:,i]*samples[:,j-1]+1
                else:
                    A_mat[i,j,:]=der_samples[:,i]*samples[:,j-1]
    return A_mat

def construct_cv_sample(A_mat,samples,mode):
    """
    @A_mat: matrix "C" in the paper (including multiple samples)
    @samples: mcmc(hmc) samples
    @mode: description of each mode could refer to the paper
    """      

    dim_var=A_mat.shape[0] #dimension of parameters
    num_var=A_mat.shape[2] #number of samples
    
    if mode==1:
        a_vec=np.zeros((num_var,dim_var))
        k=0
        for i in range(dim_var):
            a_vec[:,k]=A_mat[i,0,:]
            k+=1

    if mode==2:
        a_vec=np.zeros((num_var,2*dim_var))
        k=0
        for i in range(dim_var):
            a_vec[:,k]=A_mat[i,0,:]
            a_vec[:,k+1]=A_mat[i,i+1,:]
            k+=2  
        
    if mode==3:
        a_vec=np.zeros((num_var,(3*dim_var-1)))
        k=0
        for i in range(dim_var):
            for j in range(dim_var+1):
                if  j==0:
                    a_vec[:,k]=A_mat[i,j,:]
                    k+=1
                if  i==(j-1): 
                    a_vec[:,k]=A_mat[i,j,:]
                    k+=1
                if  i==(j-2):
                    a_vec[:,k]=A_mat[i,j,:]+A_mat[j-1,i+1,:]
                    k+=1  
                    
    if mode==4:
        a_vec=np.zeros((num_var,(4*dim_var-2)))
        k=0
        for i in range(dim_var):
            for j in range(dim_var+1):
                if  j==0:
                    a_vec[:,k]=A_mat[i,j,:]
                    k+=1
                if  i==(j-1): 
                    a_vec[:,k]=A_mat[i,j,:]
                    k+=1
                if  i==(j-2) or (i==j& j!=0):
                    a_vec[:,k]=A_mat[i,j,:]
                    k+=1  
                    
    if mode==5:       
        a_vec=np.zeros((num_var,(dim_var+1)))     
        k=0
        for i in range(dim_var+1):
            a_vec[:,k]=np.sum(A_mat[:,i,:],axis = 0)
            k+=1

    if mode==6:
        a_vec=np.zeros((num_var,int((dim_var**2+dim_var)/2)))
        k=0
        for i in range(dim_var):
            for j in range(dim_var+1):
                if  i==(j-1): 
                    a_vec[:,k]=A_mat[i,j,:]
                    k+=1
                if  i<(j-1):
                    a_vec[:,k]=A_mat[i,j,:]+A_mat[j-1,i+1,:]
                    k+=1            
                    
    if mode==7:
        a_vec=np.zeros((num_var,int((dim_var**2+3*dim_var)/2)))
        k=0
        for i in range(dim_var):
            for j in range(dim_var+1):
                if  j==0:
                    a_vec[:,k]=A_mat[i,j,:]
                    k+=1
                if  i==(j-1): 
                    a_vec[:,k]=A_mat[i,j,:]
                    k+=1
                if  i<(j-1):
                    a_vec[:,k]=A_mat[i,j,:]+A_mat[j-1,i+1,:]
                    k+=1
                       
    if mode==8:
        a_vec=np.zeros((num_var,(dim_var*(dim_var+1))))
        k=0
        for i in range(dim_var):
            for j in range(dim_var+1):
                a_vec[:,k]=A_mat[i,j,:]
                k+=1

    #optimize the coefficient
    coefficient=-np.linalg.inv(a_vec.T@a_vec)@a_vec.T@ \
        (samples-np.ones((num_var,dim_var))*np.mean(samples,axis=0))
    #output final sample by the generalized zero variance method
    g_sample=a_vec@coefficient
    cv_sample=samples+g_sample #implement "f+g"
    
    #output (samples after cv, number of coefficients optimized)
    return (cv_sample,a_vec.shape[1]) 


#calculate the variance ratio of (control variates/original)
def cv_vd(cv_sample,mcmc_samples):
    
    dim=mcmc_samples.shape[1] #dimension of parameters
    cv_vd_per=np.zeros((dim))         #(new sample variance)/(old sample variance) ratio
    for i in range(dim):        
        cv_vd_per[i]=np.var(cv_sample[:,i])/np.var(mcmc_samples[:,i])
    return cv_vd_per

