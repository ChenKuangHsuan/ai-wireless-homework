This is the code snippet of Exercise 5.10.

Environment_marl.py is the training vehicular environment, and Environment_marl_test.py is the testing vehicular environment.
marltrain.py is the MARL-DQN method proposed in L. Liang, H. Ye, and G. Y. Li, "Spectrum sharing in vehicular networks based on multi-agent reinforcement learning," IEEE Journal on Selected Areas in Communications, vol. 37, no. 10, pp. 2282-2292, Oct. 2019.
sarltrain_ppo.py is the discrete power control method discussed in Sec. 5.5.2.
sarltrain_2ppo_decay.py is the discrete spectrum access and continuous power control method.
marltest_2ppo_decay.py is the testing file.

Notably, if you want to compare discrete power control method and continuous power control method, you should keep the testing condition (i.e., channel state information) same. 
