This code tries to replicate the results obtained by Lopes, M., Lang, T., Toussaint, M., & Oudeyer, P. Y. (Neurips 2012).

The programs description is listed below : 

* to install the libraries, use requirements.txt
* to generate all the environments, launch generation_env.py 
* to do the parameter fitting, launch parameter_fitting.py
* to get the main results of the article, launch main.py
* the agents are in agents.py, the policy evaluation functions are in policy\_Functions.py, the environment is in Lopesworld.py, the important functions (such as the play function) are in main\_Functions.py 

In the folder environments, you can find all the transitions and rewards of the environments used in a .npy format. In the folder Images, you can find 2D plots of the agents performance in one environment or of the optimal value iteration in each environment. In the folder Parameter fitting, you can find the Data, the 1D plots and the heatmaps which helped us fit the parameters of the agents. In Results you can check the results that you get when you launch the main function.






