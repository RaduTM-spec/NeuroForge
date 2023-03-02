# NeuroForge
![](https://img.shields.io/badge/version-1.0.1-blue)
![](https://img.shields.io/badge/doc-paper-red)

**NeuroForge** is a an open-source framework that uses Unity as training environment and physics engine for custom 2D/3D intelligent agents created by the user. The source code provides implementations of two state-of-the-art algorithms from distinct genres with different types of neural networks required for them. Purpose of Neuroevolution of Augmenting Topologies (NEAT) resides in specific slight situations where agents can train together in the same environment but on different collision layers, whereas Proximal Policy Optimization (PPO) is more suitable for complex behaviours, moreover, where agents might need to train in different clones of the same environment, in consequence each environmental object interacts independently with each agent.

###### _**Note** - PPO is not released yet_


### Papers
###### _**Note** - Papers are not released yet_
The documentation covers the contents of use in the following chapters:
1. Installation
2. Agent initialization
3. Script overriding
4. Using NEAT
5. Using PPO
6. Reward function
7. Training environment 
8. Strategies
9. Post training

The other paper describes in depth the following:
1. Framework implementation details
2. Genetic algorithms (NEAT)
3. Deep Reinforcement Learning algorithms (PPO)

### Motivation
This framework stands as an alternative path for official **ML Agents** toolkit, simpler to understand, use and install (avoiding pytorch installation, as well as terminal interference), with genuinely potential to be a license project.

### Resources
Unity ML Agents
https://github.com/Unity-Technologies/ml-agents/tree/release_20_docs/docs/
Evolving Neural Networks through Augmenting Topologies https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
Difference-Based Mutation Operation for Neuroevolution of Augmented Topologies: 
https://iopscience.iop.org/article/10.1088/1757-899X/1047/1/012075/pdf
Proximal Policy Optimization Algorithms
https://arxiv.org/pdf/1707.06347.pdf
Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf
