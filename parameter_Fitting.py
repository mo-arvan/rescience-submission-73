import numpy as np
import time

##Optimal policies ###

"""def compute_optimal_policies(environments_parameters=environments_parameters):
    for name_environment in all_environments.keys():
        environment=all_environments[name_environment](**environments_parameters[name_environment])
        for number_world in range(1,21):
            transitions=np.load('Mondes/Transitions_'+str(number_world)+'.npy',allow_pickle=True)

            transitions_lopes=np.load('Mondes/Transitions_Lopes_non_stat'+str(number_world)+'.npy',allow_pickle=True)
            environments_parameters["Lopes_nostat_optimal_{0}".format(number_world)]={'transitions':transitions_lopes_optimal,'transitions2':transitions_lopes_i_variable}
            environments_parameters["Lopes_nostat_{0}".format(number_world)]={'transitions':transitions_lopes,'transitions2':transitions}
        gridworld=plot_VI(environment,gamma=0.95,accuracy=0.001)
        pygame.image.save(gridworld.screen,"Images/Optimal policy/VI_"+name_environment+".png")"""

### Parametter fitting ##

precision_conv=-0.2

def fitting_BEB(environment_names,betas,priors,trials = 300,max_step = 30,accuracy=0.05,screen=0,pas_VI=25,informative=True):
    BEB_parameters={(beta,prior):{'gamma':0.95,'beta':beta,'known_states':True,'coeff_prior':prior,'informative':informative} for beta in betas for prior in priors}
    pol_error={(beta,prior):[] for beta in betas for prior in priors}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for beta in betas :
            print(beta)
            for prior in priors :
                BEB=BEB_Agent(environment,**BEB_parameters[(beta,prior)]) #Defining a new agent from the dictionary agents
                
                _,step_number,policy_value_error= play(environment,BEB,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment
                pol_error[beta,prior].append(policy_value_error)

    min_length_param={(beta,prior):np.min([len(pol_error[beta,prior][i]) for i in range(len(environment_names))]) for beta in betas for prior in priors}
    mean_pol_error={(beta,prior): np.average([pol_error[beta,prior][i][:min_length_param[beta,prior]] for i in range(len(environment_names))],axis=0) for beta in betas for prior in priors}
    convergence_trial={(beta,prior):np.where(mean_pol_error[beta,prior]<precision_conv)[-1][-1] for beta in betas for prior in priors}
    convergence_step={(beta,prior):pas_VI*(convergence_trial[beta,prior]+1) for beta in betas for prior in priors}
    
    return mean_pol_error,convergence_step


def fitting_RALP(environment_names,alphas,ms,trials = 200,max_step = 30,accuracy=.05,screen=0,pas_VI=25):
    RALP_parameters={(alpha,m):{'gamma':0.95,'Rmax':1,'step_update':10,'alpha':alpha,'m':m} for alpha in alphas for m in ms}
    pol_error={(alpha,m): [] for alpha in alphas for m in ms}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for alpha in alphas :
            print(alpha)
            for m in ms :
                RALP=RmaxLP_Agent(environment,**RALP_parameters[(alpha,m)]) #Defining a new agent from the dictionary agents
                
                reward,step_number,policy_value_error= play(environment,RALP,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment
                pol_error[alpha,m].append(policy_value_error)

    min_length_param={(alpha,m):np.min([len(pol_error[alpha,m][i]) for i in range(len(environment_names))]) for alpha in alphas for m in ms}
    mean_pol_error={(alpha,m): np.average([pol_error[alpha,m][i][:min_length_param[alpha,m]] for i in range(len(environment_names))],axis=0) for alpha in alphas for m in ms}
    convergence_trial={(alpha,m):np.where(mean_pol_error[alpha,m]<precision_conv)[-1][-1] for alpha in alphas for m in ms}
    convergence_step={(alpha,m):pas_VI*(convergence_trial[alpha,m]+1) for alpha in alphas for m in ms}  
    
    return mean_pol_error,convergence_step



def fitting_QMB(environment_names,epsilons,trials=50,max_step=30,accuracy=0.01,screen=0,pas_VI=25):
    QMB_parameters={epsilon:{'gamma':0.95,'known_states':True,'epsilon':epsilon} for epsilon in epsilons}
    pol_error={epsilon:[] for epsilon in epsilons}
    
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for epsilon in epsilons:
            print(epsilon)
            QMB=QMB_Agent(environment,**QMB_parameters[epsilon]) #Defining a new agent from the dictionary agents
                
            _,step_number,policy_value_error= play(environment,QMB,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment
                    
            pol_error[epsilon].append(policy_value_error)

    min_length_param={epsilon:np.min([len(pol_error[epsilon][i]) for i in range(len(environment_names))]) for epsilon in epsilons}
    mean_pol_error={epsilon: np.average([pol_error[epsilon][i][:min_length_param[epsilon]] for i in range(len(environment_names))],axis=0) for epsilon in epsilons}
    convergence_trial={epsilon:np.where(mean_pol_error[epsilon]<precision_conv)[-1][-1] for epsilon in epsilons}
    convergence_step={epsilon:pas_VI*(convergence_trial[epsilon]+1) for epsilon in epsilons}  
    return mean_pol_error,convergence_step


def fitting_BEBLP(environment_names,betas,alphas,trials = 300,max_step = 30,accuracy=5,screen=0,pas_VI=25):
    BEBLP_parameters={(beta,alpha):{'gamma':0.95,'beta':beta,'alpha':alpha,'coeff_prior':0.001,'step_update':10} for beta in betas for alpha in alphas}
    pol_error={(beta,alpha):[] for beta in betas for alpha in alphas}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for beta in betas :
            print(beta)
            for alpha in alphas :
                BEBLP=BEBLP_Agent(environment,**BEBLP_parameters[(beta,alpha)]) #Defining a new agent from the dictionary agents
                
                _,step_number,policy_value_error= play(environment,BEBLP,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment
                pol_error[beta,alpha].append(policy_value_error)

    min_length_param={(beta,alpha):np.min([len(pol_error[beta,alpha][i]) for i in range(len(environment_names))]) for beta in betas for alpha in alphas}
    mean_pol_error={(beta,alpha): np.average([pol_error[beta,alpha][i][:min_length_param[beta,alpha]] for i in range(len(environment_names))],axis=0) for beta in betas for alpha in alphas}
    convergence_trial={(beta,alpha):np.where(mean_pol_error[beta,alpha]<precision_conv)[-1][-1] for beta in betas for alpha in alphas}
    convergence_step={(beta,alpha):pas_VI*(convergence_trial[beta,alpha]+1) for beta in betas for alpha in alphas}
    
    return mean_pol_error,convergence_step


def fitting_RA(environment_names,u_ms,ms,trials = 200,max_step = 30,accuracy=.05,screen=0,pas_VI=25):
    RALP_parameters={(m,u_m):{'gamma':0.95,'Rmax':1,'m':m,'u_m':u_m} for u_m in u_ms for m in ms}
    pol_error={(m,u_m): [] for m in ms for u_m in u_ms}
    for name_environment in environment_names:   
        print(name_environment)
        environment=all_environments[name_environment](**environments_parameters[name_environment])                
        for u_m in u_ms :
            print(u_m)
            for m in ms :
                RA=Rmax_Agent(environment,**RALP_parameters[(m,u_m)]) #Defining a new agent from the dictionary agents
                
                r_,step_number,policy_value_error= play(environment,RA,trials=trials,max_step=max_step,screen=screen,accuracy=accuracy,pas_VI=pas_VI) #Playing in environment

                pol_error[m,u_m].append(policy_value_error)
    min_length_param={(m,u_m):np.min([len(pol_error[m,u_m][i]) for i in range(len(environment_names))]) for u_m in u_ms for m in ms}
    mean_pol_error={(m,u_m): np.average([pol_error[m,u_m][i][:min_length_param[m,u_m]] for i in range(len(environment_names))],axis=0) for u_m in u_ms for m in ms}
    convergence_trial={(m,u_m):np.where(mean_pol_error[m,u_m]<precision_conv)[-1][-1] for m in ms for u_m in u_ms}
    convergence_step={(m,u_m):pas_VI*(convergence_trial[m,u_m]+1) for m in ms for u_m in u_ms} 
    return mean_pol_error,convergence_step


pas_VI=50
accuracy=0.01
trials=125
max_step=40


environment_names=['U_{0}'.format(num) for num in range(1,6)]

temps=str(time.time())
colors_6=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
markers=['^','o','x','*','s','P','.','D','1','v',',']

#E-GREEDY 

"""
epsilons=[0.005,0.1,0.2,0.5,0.8,1]
a,conv_a=fitting_QMB(environment_names,epsilons,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/e-greedy'+temps,a)
plt.figure(dpi=300)
count=0
for epsilon,mean_pol in a.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='epsilon='+str(epsilon),color=colors_6[count],marker=markers[count])
    count+=1
plt.title('ε-greedy')
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting 2/e-greedy'+temps+environment_names[0]+'.png')
plt.show()


#BEB NO PRIOR


betas=[0.5,1,2,3,4,5]
priors=[0.001]
b,conv_b=fitting_BEB(environment_names,betas=betas,priors=priors,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step,informative=False)
np.save('Parameter fitting 2/BEB no prior'+temps,b)
plt.figure(dpi=300)
count=0
for (beta,prior),mean_pol in b.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),color=colors_6[count],marker=markers[count])
    count+=1
plt.title('BEB')
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting 2/BEB no prior'+temps+'.png')
plt.show()

#BEB PRIOR

betas=[round(0.5*i,1) for i in range(1,11)]
priors=[round(0.5*i,1) for i in range(1,6)]
b,conv_b=fitting_BEB(environment_names,betas=betas,priors=priors,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/BEB'+temps,b)



markers_beta=markers[:len(betas)]
new_dict_BEB={prior:{} for prior in priors}
for (beta,prior), mean_pol in b.items():
    new_dict_BEB[prior][beta]=mean_pol

for prior in new_dict_BEB:
    plt.figure(dpi=300)
    count=0
    for beta,mean_pol in new_dict_BEB[prior].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),marker=markers_beta[count])
        count+=1
    plt.title('BEB, prior='+str(prior))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/BEB'+str(prior)+environment_names[0]+temps+'.png')
    plt.show()


ser = pd.Series(list(conv_b.values()),
                  index=pd.MultiIndex.from_tuples(conv_b.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('Priors')
plt.ylabel('Betas')
plt.title('Pas de convergence BEB')
plt.savefig('Parameter fitting 2/BEB_heatmap'+temps+environment_names[0]+'.png')
plt.show()

#RALP

alphas=[round(0.5*i,1) for i in range(1,6)]
ms=[round(0.5*i,1) for i in range(1,10)]
c,conv_c=fitting_RALP(environment_names,alphas=alphas,ms=ms,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/RALP'+temps,c)


markers_m=markers[:len(ms)]
new_dict_RALP={alpha:{} for alpha in alphas}
for (alpha,m), mean_pol in c.items():
    new_dict_RALP[alpha][m]=mean_pol



for alpha in new_dict_RALP:
    plt.figure(dpi=300)
    count=0
    for m,mean_pol in new_dict_RALP[alpha].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='m='+str(m),marker=markers_m[count])
        count+=1
    plt.title('RALP, alpha='+str(alpha))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/RALP'+str(alpha)+environment_names[0]+temps+'.png')
    plt.show()


ser = pd.Series(list(conv_c.values()),
                  index=pd.MultiIndex.from_tuples(conv_c.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('m')
plt.ylabel('alpha')
plt.title('Pas de convergence RmaxLP')
plt.savefig('Parameter fitting 2/RALP_heatmap'+environment_names[0]+temps+'.png')
plt.show()

plt.figure(dpi=300)
count=0
for (alpha,m),mean_pol in c.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='m='+str(m),color=colors_6[count],marker=markers[count])
    count+=1
plt.title('Rmax-LP α=0.5')
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting/RmaxLP'+temps+'.png')
plt.show()"""
"""
#BEBLP

alphas=[round(0.5*i,1) for i in range(1,6)]
betas=[round(0.5*i,1) for i in range(1,11)]

d,conv_d=fitting_BEBLP(environment_names,alphas=alphas,betas=betas,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/BEBLP'+temps,d)


markers_beta=markers[:len(betas)]
new_dict_BEBLP={alpha:{} for alpha in alphas}
for (beta,alpha), mean_pol in d.items():
    new_dict_BEBLP[alpha][beta]=mean_pol



for alpha in new_dict_BEBLP:
    plt.figure(dpi=300)
    count=0
    for beta,mean_pol in new_dict_BEBLP[alpha].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),marker=markers_beta[count])
        count+=1
    plt.title('BEBLP, alpha='+str(alpha))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/BEBLP'+str(alpha)+environment_names[0]+temps+'.png')
    plt.show()


ser = pd.Series(list(conv_d.values()),
                  index=pd.MultiIndex.from_tuples(conv_d.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.title('Pas de convergence BEBLP')
plt.savefig('Parameter fitting 2/BEBLP_heatmap'+environment_names[0]+temps+'.png')
plt.show()


plt.figure(dpi=300)
count=0
for (beta,alpha),mean_pol in d.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),color=colors_6[count],marker=markers[count])
    count+=1
plt.title('BEB-LP')
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting/BEBLP'+temps+'.png')
plt.show()
"""
#Rmax
"""
ms=[i for i in range(1,11)]
u_ms=[i for i in range(1,30,3)]
markers_m=markers[:len(ms)]
e,conv_e=fitting_RA(environment_names,u_ms=u_ms,ms=ms,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/Rmax'+temps,e)

new_dict={u_m:{} for u_m in u_ms}
for (m,u_m), mean_p in e.items():
    new_dict[u_m][m]=mean_p

for u_m in new_dict:
    plt.figure(dpi=300)
    count=0
    for m,mean_pol in new_dict[u_m].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='m='+str(m),marker=markers_m[count])
        count+=1
    plt.title('Rmax, u_m='+str(u_m))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/Rmax'+str(u_m)+environment_names[0]+temps+'.png')
    plt.show()

ser = pd.Series(list(conv_e.values()),
                  index=pd.MultiIndex.from_tuples(conv_e.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('u_m')
plt.ylabel('m')
plt.title('Pas de convergence Rmax')
plt.savefig('Parameter fitting 2/Rmax_heatmap'+environment_names[0]+temps+'.png')
plt.show()



plt.figure(dpi=300)
count=0
for (m,u_m),mean_pol in e.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='m='+str(m),color=colors_6[count],marker=markers[count])
    count+=1
plt.title('Rmax')
plt.xlabel("Steps")
plt.ylabel("Policy value error")
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting/Rmax'+temps+'.png')
plt.show()
"""

"""
gamma_es=[0.1*i for i in range(6)]
coeff_es=[1*i for i in range(6)]
f,conv_f=fitting_EGE(environment_names,gamma_es=gamma_es,coeff_es=coeff_es,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting/EGE'+temps,f)

plt.figure(dpi=300)
for (gamma_e,coeff_e),mean_pol in f.items() : 
    plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='gamma_e='+str(gamma_e)+', coeff_e='+str(coeff_e))
plt.title('EGE')
plt.grid(linestyle='--')
plt.legend()
plt.savefig('Parameter fitting/EGE'+environment_names[0]+temps+'.png')
plt.show()

ser = pd.Series(list(conv_f.values()),
                  index=pd.MultiIndex.from_tuples(conv_f.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('gamma_e')
plt.ylabel('coeff_e')
plt.title('Pas de convergence ε-greedy-E')
plt.savefig('Parameter fitting/EGE_heatmap'+environment_names[0]+temps+'.png')
plt.show()

# PIM    
    

alphas=[10**(i) for i in range(-3,4)]
betas=[10**(i) for i in range(-3,4)]
g,conv_g=fitting_PIM(environment_names,alphas=alphas,betas=betas,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/PIM'+temps,g)


new_dict_PIM={alpha:{} for alpha in alphas}
for (alpha,beta), mean_p in g.items():
    new_dict_PIM[alpha][beta]=mean_p


for alpha in new_dict_PIM:
    plt.figure(dpi=300)
    count=0
    for beta,mean_pol in new_dict_PIM[alpha].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),marker=markers[count])
        count+=1
    plt.title('PIM, alpha='+str(alpha))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/PIM'+str(alpha)+environment_names[0]+temps+'.png')
    plt.show()


ser = pd.Series(list(conv_g.values()),
                  index=pd.MultiIndex.from_tuples(conv_g.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('beta')
plt.ylabel('alpha')
plt.title('Pas de convergence PIM')
plt.savefig('Parameter fitting 2/PIM_heatmap'+environment_names[0]+temps+'.png')
plt.show()
"""

"""
alphas=[10**(i) for i in range(-3,4)]
betas=[10**(i) for i in range(-3,4)]
g,conv_g=fitting_PIM(environment_names,alphas=alphas,betas=betas,pas_VI=pas_VI,accuracy=accuracy,trials=trials,max_step=max_step)
np.save('Parameter fitting 2/PEI-DE'+temps,g)


new_dict_PIM={alpha:{} for alpha in alphas}
for (alpha,beta), mean_p in g.items():
    new_dict_PIM[alpha][beta]=mean_p


for alpha in new_dict_PIM:
    plt.figure(dpi=300)
    count=0
    for beta,mean_pol in new_dict_PIM[alpha].items():
        plt.plot([pas_VI*i for i in range(len(mean_pol))],mean_pol,label='beta='+str(beta),marker=markers[count])
        count+=1
    plt.title('PIM, alpha='+str(alpha))
    plt.grid(linestyle='--')
    plt.legend()
    plt.savefig('Parameter fitting 2/PEI-DE'+str(alpha)+environment_names[0]+temps+'.png')
    plt.show()


ser = pd.Series(list(conv_g.values()),
                  index=pd.MultiIndex.from_tuples(conv_g.keys()))
df = ser.unstack().fillna(0)
df.shape
plt.figure()
ax=sns.heatmap(df,cmap='Blues')
plt.xlabel('beta')
plt.ylabel('alpha')
plt.title('Pas de convergence PEI-DE')
plt.savefig('Parameter fitting 2/PEI-DE_heatmap'+environment_names[0]+temps+'.png')
plt.show()"""