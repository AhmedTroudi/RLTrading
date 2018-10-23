# RLTrading
Algorithm Trading in RL using Q learning Algorithm

# Reinforcement Learning
This area of machine learning consists in training an agent by reward and punishment without needing to
specify the expected action. The agent learns from its experience and develops a strategy that maximizes its
profits. The goal is to come up with a profitable strategy based on a single parameter which is price prediction
by trading a single stock under the reinforcement learning framework.

# Why choose RL?
Here’s a list :
• Instead of needing to hand-code a rule-based policy which requires a certain level of expertise in finance,
Reinforcement Learning directly learns a policy. There’s no need for us to specify rules and thresholds
such as “buy when you are more than 75% sure that the market will move up”. That’s baked in the
RL policy, which optimizes for the metric we care about. We’re removing a full step from the strategy
development process!
• Because the policy can be parameterized by a complex model, such as a Deep Neural network, we can
learn policies that are more complex and powerful than any rules a human trader could possibly come up
with.
• Because RL agents are learning powerful policies parameterized by Neural Networks, they can also learn
to adapt to various market conditions by seeing them in historical data, given that they are trained over
a long time horizon and have sufficient memory. This allows them to be much more robust to changing
markets.
