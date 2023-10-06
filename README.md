Reinforcement Learning for Trading Cryptocurrencies

Henrique Magalhães Rio and Mark Gardner
DSCI-478
05/04/2022



Introduction
	Reinforcement learning is a type of deep learning algorithm that teaches programs to take actions within a defined environment. It became famous through the application of algorithms that could learn to strategically play games such as chess, Atari games, and other strategic games. Currently, reinforcement learning (RL) is heavily used in the auto industry for the development of self-driving cars. In essence, RL works well in applications in which there are “rules” that can be defined. In the example of a self-driving car, they are well-defined traffic rules (i.e. stop signs, don’t crash, stay in lane, stop at a red light, etc), and so the model can be trained based on these rules (Krohn et al., 2019, p. 287).
	In our project, we decided to use RL for automating cryptocurrency trading as it seems to be an application that translates well within the RL framework. At the core of reinforcement is the idea of maximizing a return function based on the reward of each action done by the agent. In the trading problem, for example, those can be defined quite easily since the actions are just buying, holding, or selling, and the return is simply how much money a trading strategy made over a given period. In our project, we are using a package called FinRl from the Ai4Finance foundation in order to implement the algorithms using cryptocurrency data from Yahoo Finance.

Reinforcement Learning
The reinforcement learning cryptocurrency environment can be thought of as our portfolio/trading area as from our portfolio our agent can obtain the state (St) of the environment which are: current prices , how many of each coin it currently owned, how much money it has spent and can still spend, and how much the portfolio is currently worth at any given time. In the environment, the actions (at) the agent can take to buy a cryptocurrency, hold the currency, or sell the currency. We can then calculate the reward of the action by looking at the change in the state (Portfolio value) using a reward function rt=Rt(st) (Part 1: Key Concepts in RL — Spinning Up Documentation, n.d.). With the reward defined we can start to define the agent optimization problem, in which our agent follows a policy π which is looking at the current state of the market and executing an action (Buying,Holding,Selling). Therefore, we want to optimize the policy so that the sum of all rewards, which is the cumulative returns, is maximized (Part 1: Key Concepts in RL — Spinning Up Documentation, n.d.).
Moreover, the RL optimization can be summarized by the Optimal Action-Value function:
Q*(s,a)= maxE~[R(|s0=s,a0=a']
The optimal Q value function returns the expected returns given a start in state s and an action a, which is calculated by taking the optimized policy π, and sampling the trajectory 𝝉 from that policy so that we can obtain the the reward for each state and action in said trajectory.(spinningup.openai.com)  In our crypto problem, we train our agent by having it follow different policies (trading strategies) in which it buys, sells or holds a cryptocurrency with the goal of maximizing the cumulative return over the whole period of time.  Moreover, the reward function is usually multiplied by a discount factor in order to ensure that a lower reward over a shorter period of time is better than a higher reward over a longer period of time (Part 1: Key Concepts in RL — Spinning Up Documentation, n.d.). 


Reinforcement Learning Algorithms

 	The taxonomy of RL Algorithms is depicted above.  Our research focused on both types of model-free RL algorithms shown here, being Policy Optimization and Q-Learning..  Though before describing model-free learning, it’s worth clarifying the difference between the two types of reinforcement learning algorithms. 
 
Model-based algorithms have access to/or learn from a model of their environment.   The models are actually functions that predict the state transitions and rewards for the agent (policy maximizer). The primary advantage of these models is their ability to see a range of potential choices, to decide between available options, then to synthesize the results into an informed policy for the future.  Often, a true model for the environment isn’t available, so an agent (policy maximizer) must learn a model from the available environment. This creates a potential for the agent to exploit in-sample bias, leading to over-fitting on training data and underperforming in out-of-sample tests. 
 
There are two main fields of model-free RL learning: Policy Optimization and Q-learning.  Two of our tested algorithms: A2C and PPO, are policy optimizers.  While the remainder of our tested algorithms: DDQN, DQN and SAC, are Q-Learning or a combination of the two.  To distinguish between the two, we introduce the idea of being on-policy (Policy Optimization) vs off-policy (Q-Learning).  The primary difference is that, in each update of the replay buffer, off-policy algorithms can use data gained throughout training, independent of how the agent acted when it first encountered the data, whereas on-policy means that each update can only use data gained after acting in accordance with the most recent iteration of the policy. 
 
Where P.O. algorithms are stable and reliable because they ‘directly optimize for the thing you want,’ Q-Learning algorithms indirectly optimize for agent performance “by training Qθ to satisfy a self-consistency equation” (Selecting an Algorithm, 2022), which causes more points of failure and higher instability).  Despite these drawbacks, because they can reuse data better than P.O., Q-Learning methods are significantly more efficient at sampling, when they do work. 
 
There exists a third type of algorithms not explicitly discussed yet: combinations of P.O. and Q-Learning . DDPG and SAC are two examples used in our research.   It is with these types of algorithms that our discussion of the specific implementation of these R.L. algorithms now begins.  The next section will explore the elements of each of the algorithms we used. 
https://intellabs.github.io/coach/selecting_an_algorithm.html
 
PPO algorithms use an actor/critic scheme that creates a stable learning process by bounding the updates to the policy (Schulman, 2017).  The distinguishing feature of this type of algorithms is that they switch from sampling data by interacting with its environment to using stochastic gradient ascent to optimize a ‘surrogate’ objective function.  PPO utilizes a new objective function that allows for multiple epochs of mini-batch updates, while normal policy gradient methods are constrained to a single gradient update per sample. PPO receives some of the benefits of TRPOs (trust region policy optimization algorithms) yet are simpler to implement.  Of note, when tested against various benchmark tests, PPO struck a reasonable balance between wall-time, simplicity, and sample complexity (identify these metrics). 
 
A2C, or Advantage Actor Critic, is a combination of two types of policy based and value-based reinforcement learning algorithms (Wang, 2021). Again, while value-based algorithms deterministically select from possible actions based on the predicted value of the input state or action, policy based algorithms map input states to output actions after learning a probability distribution of actions, or a policy.  
A2C is a version of the A3C policy gradient method.  A2C is deterministic, synchronous, and waits for both actors to finish their segments of experience before updating and averaging over both the actors.  One of the benefits of this method is that it is able to use GPU’s more efficiently through parallel learning, which allows multiple workers to decrease data collection time and improve speed and stability in discrete and continuous action spaces  (Papers with code, 2022). 
 
DDPG is an off-poilcy, model-free, actor-critic scheme for continuous action spaces which assumes that the policy is deterministic, and it uses a replay buffer in order to improve sample efficiency (Deep deterministic policy gradient, 2022).  It is helpful to think of DDP as a deep Q-learning algorithm for continuous action spaces. It simultaneously learns a policy and Q-function, first by using the Bellman equation and off-policy data to learn the Q-function, then by using that function to learn the policy.  
 
While the task of solving a continuous optimization problem is normally computationally prohibitive, DDPG leverages the fact that the function (Q*(s,a)) is assumed to be differentiable with respect to the action, which allows for the creation of an efficient, gradient-based learning for the policy.  Instead of using costly optimization each time max_a Q(s,a) is computed, it can be approximated by DDPG more efficiently. 
 
TD3 is similar to DDPG, except that it is more resilient when tuning hyperparameters.  Where DDPG often fails as a result of dramatically overestimating Q-values by the Q-function, TD3 confronts this core problem in three ways.
 
First, with Clipped Double-Q Learning, TD3 uses two Q-functions, as opposed to one, and chooses the smaller Q-value to create the Bellman error loss functions.  Then, it implements delayed policy updates, meaning that it updates the target networks and policy less often.  Lastly, it adds noisy data  to the target action, smoothing out the Q function along changes in its action, making it more difficult for the policy to exploit Q-function errors (Twin delayed DDPG, 2022). 
 
DDPG has an actor-critic in continuous action spaces. It uses a replay buffer in order to improve sample efficiency; it uses two critic networks in order to mitigate the overestimation in the Q state-action value prediction; and, it slows down the actor updates in order to increase stability and adds noise to actions while training the critic in order to smooth out the critic’s predictions. (this feels like I’m saying the same thing twice). 
 
Soft-Actor-Critic (SAC) optimizes a stochastic policy in an off-policy way (Hararnoja, 2018).   It is a model-free, deep RL algorithm.  Two challenges occur often in model-free algorithms: brittle convergence properties and high sample complexity. SAC is based on a maximum entropy reinforcement learning framework.  In short, in such frameworks the actor seeks to maximize expected rewards and entropy..  In this instance, maximizing entropy would mean allowing the actor to behave as randomly as possible. 
 
SAC combines a stable stochastic actor-critic pairing with off-policy updates, allowing it to address and improve upon  the instability common in many off-policy algorithms.
 
The decision about which of these algorithms to use, and when to use them, can be found after asking the following questions. Is it a discrete or continuous action space? Do you have expert demonstrations for your task?  Can you collect new data for your task dynamically?  Do you have a simulator for your task? The included citation contains an interactive tool that filters algorithms based on these conditions which proved useful to our work (Selecting an Algorithm, 2018).  
Results
In order to test the different kinds of algorithms we trained and tested their results in the same period and using the same cryptocurrencies which are ADA, BCH, BTC, BNB, DOGE, ETC, ETH, LINK, LTC, TRX, and XRP. These cryptocurrencies were choosen most due to data avalaibility, since some of the cryptocurrencies new and therefore, did not have data on the periods choose for training and testing which were, 2016-01-01 to 2020-06-01 for the training, and  2020-06-01 to 2022-04-8 for the testing. In order to compare our results we used the S&P500 index as baseline which is a standard for the financial industry since it is a index that constantly performs really well, all algorithms started with the same amount of cash to invest which was 1 million U.S. dollars.


A2C





Advanced Actor Critic (A2C) is the first model we tested, which seems to take a very conservative approach to trading cryptocurrencies, the strategy that it took in our test  consisted of choosing a few cryptocurrencies (BTC, ETH, LINK TRX, and XRP) buying them and holding those until the end of the period. In the case of the crypto currencies this strategy seems to have worked well as we can see by the Figure X (new normal) it is well above the baseline strategy of buying and holding the S&P500.
DDPG

For the DDPG we can see that it again picked a few cryptocurrencies, however, these are different from the ones that A2C choose as it picked ADA, BCH, DOGE, LINK, and  XRP. As we can see it well in the beginning of the training data, however, around the middle of 2021 it started incurring big losses, which lead to annual return of -20% in 2022.













PPO


	Lastly we have PPO which seems to go a completely different route when compared to the other 2 algorithms, again it did choose a few cryptocurrencies to trade which were ADA, BNB, BTC, DOGE, ETH, LTC, TRX, and XRP. However, once it choose it did execute quite a few transactions on those coins, as we can see from figure X below, it bought and sold BTC at various points with a total of 297 transactions. It seems that it worked pretty well, as from figure X (newnormal) it did a lot better when compared to the S&P500 index.


Conclusion
	After testing all of the algorithms it seems that PPO had the highest overall returns, it also had a strategy that was different from DDPG and A2C, which likely led it to have better returns. Moreover, I think the choice of the cryptocurrencies was also of great importance in the long term profit of the algorithms A2C seemed to choose some of the more expensive coins that are more established such as BTC and ETH, which presented less risk despite the upfront cost. Meanwhile, DDPG seemed to go with less upfront cost in the cheaper coins such as DOGE and ADA which are a lot more volatile and as we can see it did not pay off as the algorithm ended up losing money. PPO seemed to go with a mix between the more established currencies (BTC) and less established cryptocurrencies (DOGE), which lead to massive returns especially since it seemed to do much better at offloading some the cryptocurrencies at the right time. 
Overall, our approach in using RL for cryptocurrency trading does not seem feasible for real world application, this is mostly due to the nature of the crypto market which seems much more susceptible for volatility which make it hard to come up with efficient strategies. Also, the cryptomarket is newer when compared to the stock market which means that there is not only a lot less data to train or models in, but also the it can and it has changed lot since it has started therefore, training the model in the past may not be useful for the future due to new trends arising.



































Code
https://github.com/henriquem27/crypto-dsci478

References
 
Deep deterministic policy gradient¶. Deep Deterministic Policy Gradient - Spinning   Up documentation. (n.d.). Retrieved May 2, 2022, from https://spinningup.opena	i.com/en/latest/algorithms/ddpg.html 
Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018, August 8). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv.org. Retrieved May 2, 2022, from https://arxiv.org/abs/1801.01290 
Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018, August 8). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. arXiv.org. Retrieved May 2, 2022, from https://arxiv.org/abs/1801.01290 
Papers with code - a2c explained. Explained | Papers With Code. (n.d.). Retrieved May 2, 2022, from https://paperswithcode.com/method/a2c 
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017, August 28). Proximal policy optimization algorithms. arXiv.org. Retrieved May 2, 2022, from https://arxiv.org/abs/1707.06347 
Selecting an algorithm¶. Selecting an Algorithm - Reinforcement Learning Coach 0.12.0 documentation. (n.d.). Retrieved May 2, 2022, from https://intellabs.github.io/coach/selecting_an_algorithm.html 
Twin delayed DDPG¶. Twin Delayed DDPG - Spinning Up documentation. (n.d.). Retrieved May 2, 2022, from https://spinningup.openai.com/en/latest/algorithms/td3.html 
Wang, M. (2021, January 26). Advantage actor critic tutorial: MINA2C. Medium. Retrieved May 2, 2022, from https://towardsdatascience.com/advantage-actor-critic-tutorial-mina2c-7a3249962fc8
Part 1: Key Concepts in RL — Spinning Up documentation. (n.d.). Spinning Up in Deep RL! Retrieved May 3, 2022, from https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
Krohn, J., Bassens, A., & Beyleveld, G. (2019). Deep Learning Illustrated: A Visual, Interactive Guide to Artificial Intelligence. Addison Wesley.
(n.d.). Welcome to FinRL Library! — FinRL 0.3.1 documentation. Retrieved May 3, 2022, from https://finrl.readthedocs.io/en/latest/

