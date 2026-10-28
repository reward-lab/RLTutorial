---
layout: demo
scene: part1
cover: true
part: Part 1 · Key Concepts
---

<div class="kicker">Part 1</div>

# Key Concepts in RL

States, actions, policies, returns and value functions: the vocabulary of reinforcement learning.

<div class="meta">spinningup.openai.com</div>

<!--
This robot acts completely at random. By the end of Part 1 we will be able to say precisely how bad that is, and what the best policy looks like.
-->

---
layout: demo
scene: what-can-rl-do
kicker: Part 1
---

# What can RL do?

::left::

<v-clicks>

- Control robots, in simulation and in the real world.
- Beat top human players at strategy games: Go and Dota.
- Play Atari games straight from raw pixels.
- Teach simulated robots to follow human instructions.

</v-clicks>

<!--
These are Spinning Up's opening examples. The robot on the right starts knowing nothing and improves by trial and error (a simple actor-critic learner, a close cousin of the policy gradient in Part 3). Every success on the left is this same loop at a much larger scale.
-->

---
layout: demo
scene: meet
kicker: The running example
---

# Meet the gridworld

::left::

<v-clicks>

- A robot starts in the bottom-left corner.
- Goal: <b class="pos">+10</b>. Lava: <b class="neg">−50</b>. Either one ends the episode.
- Every other step costs <b class="neg">−1</b>, so shorter paths are better.
- The floor is slippery: sometimes the robot slides sideways instead.

</v-clicks>

<!--
Point at each element on the grid as you reveal it. Slip = probability of sliding sideways; we will use it to make the environment random.
-->

---
layout: demo
scene: loop
kicker: Key concepts
---

# Agent and environment

::left::

<v-clicks>

- The agent observes the state and picks an action.
- The environment moves to a new state and hands back a reward.
- The agent's goal: maximize its cumulative reward, the **return**.

</v-clicks>

<!--
The robot is the agent. Everything else (walls, lava, goal, the rules of motion) is the environment. This agent acts at random. It has no idea what it is doing yet; the rest of the lecture is about fixing that.
-->

---
layout: demo
scene: states
kicker: Key concepts
---

# States and observations

::left::

<v-clicks>

- State $s$: a complete description of the world.
- Here $s$ = (row, column): the map never changes, so position is everything.
- Observation $o$: what the agent sees. Our agent sees the full state, $o = s$.

</v-clicks>

<!--
The walls, lava and goal are fixed parts of the environment, so the agent's position is the whole state. In general an agent may see only part of the state; we assume full observability throughout.
-->

---
layout: demo
scene: actions
kicker: Key concepts
---

# Action spaces

::left::

<v-clicks>

- The action space is the set of all valid actions.
- $$
  \begin{array}{l}
  \text{discrete: } a \in \{\uparrow, \rightarrow, \downarrow, \leftarrow\} \\
  \text{continuous: } a \in \mathbb{R}^2
  \end{array}
  $$
- Our robot has 4 moves. A robot arm outputs a real-valued vector.
- Some algorithms only work with one of the two.

</v-clicks>

<!--
Flip to "Continuous" under the grid: now the agent picks a 2-D velocity. The yellow arrow is the action taken; faint arrows are other actions it could have sampled.
-->

---
layout: demo
scene: policies
kicker: Key concepts
---

# Policies

::left::

<v-clicks>

- A policy is the agent's rule for choosing actions.
- $$
  \begin{array}{l}
  \text{deterministic: } a_t = \mu(s_t) \\
  \text{stochastic: } a_t \sim \pi(\cdot \mid s_t)
  \end{array}
  $$
- In deep RL the policy is a neural network with weights $\theta$: $\pi_\theta$.

</v-clicks>

<!--
The arrows in every cell ARE the policy. Switch to Stochastic and drag the temperature from 0.1 to 5: near-deterministic blurs into uniform random. The bars show π(·|s) at the robot's cell; yellow is the sampled action.
-->

---
layout: demo
scene: stochastic
kicker: Key concepts
---

# Stochastic policies in practice

::left::

<v-clicks>

- Categorical, for discrete actions: the network outputs one logit per action; softmax turns them into probabilities.
- $$
  \pi_\theta(\cdot \mid s) = \operatorname{softmax}(\text{logits}), \quad a \sim \pi_\theta(\cdot \mid s)
  $$
- Diagonal Gaussian, for continuous actions: the network outputs a mean $\mu_\theta(s)$; $\log\sigma$ is learned too.
- $$
  a = \mu_\theta(s) + \sigma_\theta(s) \odot z, \quad z \sim \mathcal{N}(0, I)
  $$

</v-clicks>

<!--
A stochastic policy is a distribution we can sample from. Your notebook's Categorical(logits=logits_net(obs)) is the first case. Here the logits are Q*(s,a)/T. Switch to Gaussian to see μ, σ and the noise z behind each sampled 2-D action. log σ is learned rather than σ because it can be any real number.
-->

---
layout: demo
scene: trajectories
kicker: Key concepts
---

# Trajectories

::left::

<v-clicks>

- A trajectory is the sequence of states and actions in an episode.
- $$
  \begin{array}{l}
  \tau = (s_0, a_0, s_1, a_1, \dots) \\
  s_{t+1} \sim P(\cdot \mid s_t, a_t)
  \end{array}
  $$
- Same policy, random transitions: different trajectories every time.

</v-clicks>

<!--
The policy here is deterministic, so all the randomness comes from the slippery floor. Set slip to 0: every trajectory is identical. Raise it: the grey paths fan out. Red chips are slips.
-->

---
layout: demo
scene: return
kicker: Key concepts
---

# Reward and return

::left::

<v-clicks>

- $$
  r_t = R(s_t, a_t, s_{t+1})
  $$
- The return adds up the rewards of a whole trajectory.
- $$
  \begin{array}{ll}
  R(\tau) = \sum_{t=0}^{T} r_t & \text{\small finite horizon} \\[4pt]
  R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t & \text{\small discounted}
  \end{array}
  $$
- $\gamma < 1$: a reward now is worth more than the same reward later.

</v-clicks>

<!--
Lower γ toward 0 and watch the γ^t column collapse: far-future rewards (like the +10 goal) barely count, so the agent becomes short-sighted. Finite-horizon mode cuts the episode at T steps.
-->

---
layout: demo
scene: problem
kicker: Key concepts
---

# The RL problem

::left::

<v-clicks>

- The return is random: it depends on the policy and on the environment.
- So we maximize the **expected** return:
- $$
  \begin{array}{l}
  J(\pi) = \mathbb{E}_{\tau\sim\pi}\big[R(\tau)\big] \\
  \pi^* = \arg\max_\pi J(\pi)
  \end{array}
  $$

</v-clicks>

<!--
Each tick runs a dozen episodes. Blue shading = where the agent spends time. The yellow line (sample average) converges to the dashed line (exact J). Compare Random vs Greedy.
-->

---
layout: demo
scene: values
kicker: Key concepts
---

# Value functions

::left::

<v-clicks>

- How good is it to be here, if I follow $\pi$ from now on?
- $$
  \begin{array}{l}
  V^\pi(s) = \mathbb{E}_{\tau\sim\pi}\big[R(\tau) \mid s_0 = s\big] \\
  Q^\pi(s,a) = \mathbb{E}_{\tau\sim\pi}\big[R(\tau) \mid s_0 = s, a_0 = a\big]
  \end{array}
  $$
- $Q$ fixes the first action, then follows $\pi$.

</v-clicks>

<!--
Play runs iterative policy evaluation: each sweep spreads value one more step out from the goal and the lava. Random π: cells near lava are badly negative. Switch to Greedy: the whole map turns positive. Toggle Q to see four triangles per cell.
-->

---
layout: demo
scene: optimal
kicker: Key concepts
---

# Optimal values and the optimal action

::left::

<v-clicks>

- Same questions, under the best possible policy.
- $$
  \begin{array}{l}
  V^*(s) = \max_\pi V^\pi(s) \\
  Q^*(s,a) = \max_\pi Q^\pi(s,a)
  \end{array}
  $$
- Know $Q^*$ and acting optimally is easy:
- $$
  a^*(s) = \arg\max_a Q^*(s,a)
  $$

</v-clicks>

<!--
Play runs value iteration; the arrows are the greedy action under the current estimate, so the optimal policy emerges as values settle. Raise slip to 0.4: the route starts avoiding cells next to lava, even if longer. In the Q* view, the circle marks the best action in each cell.
-->

---
layout: demo
scene: bellman
kicker: Key concepts
---

# Bellman equations

::left::

<v-clicks>

- Value here = reward now + discounted value of where you land.
- $$
  \begin{array}{l}
  V^\pi(s) = \mathbb{E}_{a\sim\pi,\, s'\sim P}\big[r + \gamma V^\pi(s')\big] \\
  V^*(s) = \max_a \mathbb{E}_{s'\sim P}\big[r + \gamma V^*(s')\big]
  \end{array}
  $$
- $\pi$: average over actions. $*$: best action.

</v-clicks>

<!--
Click any cell on the grid. The animation cycles through the four actions and draws where each one can land, with probabilities. The panel works the equation with the real numbers; the total matches the number in the cell.
-->

---
layout: demo
scene: advantage
kicker: Key concepts
---

# Advantage functions

::left::

<v-clicks>

- How much better is action $a$ than what $\pi$ does on average?
- $$
  A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
  $$
- Green: better than average. Red: worse.
- Policy gradients raise $\pi(a \mid s)$ for actions with positive advantage.

</v-clicks>

<!--
Under the random policy, advantages point toward the goal and away from lava: that is the learning signal. Switch to Greedy: the chosen action has advantage 0 and the rest are negative. Nothing left to improve. Click a cell for its numbers.
-->

---
layout: demo
scene: formal
kicker: Key concepts
---

# The formal picture

::left::

<v-clicks>

- A Markov decision process: $\langle S, A, R, P, \rho_0 \rangle$
- Markov property: the next state depends only on the current state and action, not on the history.
- $$
  \begin{array}{l}
  P(\tau \mid \pi) = \rho_0(s_0) \prod_t P(s_{t+1} \mid s_t, a_t)\, \pi(a_t \mid s_t) \\
  J(\pi) = \int_\tau P(\tau \mid \pi) R(\tau) = \mathbb{E}_{\tau\sim\pi}\big[R(\tau)\big]
  \end{array}
  $$
- $$
  \begin{array}{l}
  V^\pi(s) = \mathbb{E}_{a\sim\pi}\big[Q^\pi(s,a)\big] \\
  V^*(s) = \max_a Q^*(s,a)
  \end{array}
  $$

</v-clicks>

<!--
Map each symbol onto the grid: S = the 22 open cells, A = 4 moves, R = −1/+10/−50, P = the slip rule, ρ₀ = always the start cell. The live readout multiplies π(a|s)·P(s′|s,a) for every step of the current episode: the probability of this exact trajectory shrinks fast, which is why we work with log-probabilities.
-->
