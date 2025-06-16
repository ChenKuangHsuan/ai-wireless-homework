This is the code snippet of Exercise 5.14.

Requirement: Python > 3.6, torch > 1.4.0

Environment_marl: The environment of a single task for learning from scratch or fine tuning.
Environment_meta: The environment, of which the five key factors can be changed to generate different tasks.
sarltrain_ppo_adapt: The training code for a single task. It can be used to learn from scratch and fine-tune the meta parameters.
metatrain_ppo_meta: The code for meta training.

The whole process:

1. Run the metatrain_ppo_meta.py to obtain the meta parameters.
2. Use the meta parameters as the initialization to adapt to a new environment.

Notably, this is an example that uses Reptile algorithm to improve generalization. You can refer to this code snippet and replace the meta learning algorithms for training.
