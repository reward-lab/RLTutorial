---
layout: demo
scene: part3
cover: true
part: Part 3 · Intro to Policy Optimization
---

<div class="kicker">Part 3</div>

# Intro to Policy Optimization

Derive the policy gradient, implement it, then make it learn faster.

<div class="meta">spinningup.openai.com</div>

<!--
Part 1 gave us J(π) and the advantage. Now we improve a policy directly by gradient ascent on J. From here on, returns are finite-horizon and undiscounted (episodes capped at 50 steps), as in Spinning Up Part 3.
-->

---
layout: demo
scene: ascent
kicker: Part 3
---

# Gradient ascent on the policy

::left::

<v-clicks>

- Parameterize the policy as $\pi_\theta$. Here $\theta$ is one logit per cell and action: 22 × 4 numbers.
- $$
  \theta_{k+1} = \theta_k + \alpha \nabla_\theta J(\pi_\theta)
  $$
- $J$ uses the finite-horizon, undiscounted return: $R(\tau) = \sum_t r_t$.
- The catch: $J$ is an expectation over trajectories. How do we take its gradient?

</v-clicks>

<!--
The arrows are softmax(θ) in every cell, starting uniform. Press play: θ climbs J and the arrows sharpen toward the goal. The rest of Part 3 explains the update that makes this happen.
-->

---
layout: demo
scene: ingredients
kicker: Part 3
---

# Deriving the policy gradient: the ingredients

::left::

<v-clicks>

- $$
  P(\tau \mid \theta) = \rho_0(s_0) \prod_t P(s_{t+1} \mid s_t, a_t)\, \pi_\theta(a_t \mid s_t)
  $$
- Log-derivative trick: $\nabla_\theta P(\tau \mid \theta) = P(\tau \mid \theta)\, \nabla_\theta \log P(\tau \mid \theta)$
- $$
  \begin{array}{l}
  \log P(\tau \mid \theta) = \log \rho_0(s_0) \\[2pt]
  \quad + \sum_t \big[\log P(s_{t+1} \mid s_t, a_t) + \log \pi_\theta(a_t \mid s_t)\big]
  \end{array}
  $$
- The environment terms don't depend on $\theta$, so their gradient is zero:
- $$
  \nabla_\theta \log P(\tau \mid \theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
  $$

</v-clicks>

<!--
Walk through the facts one by one. The table splits the current trajectory's log-probability into the part we control (policy, blue) and the part we don't (environment, grey). Only the blue part changes with θ, which is why we never need to know P.
-->

---
layout: demo
scene: result
kicker: Part 3
---

# Deriving the policy gradient: the result

::left::

<v-clicks>

- $$
  \begin{array}{l}
  \nabla_\theta J = \int_\tau \nabla_\theta P(\tau \mid \theta) R(\tau) \\[2pt]
  \phantom{\nabla_\theta J} = \mathbb{E}_{\tau\sim\pi_\theta}\big[\nabla_\theta \log P(\tau \mid \theta) R(\tau)\big]
  \end{array}
  $$
- $$
  \nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R(\tau)\Big]
  $$
- It's an expectation, so we estimate it from a batch $\mathcal{D}$ of sampled trajectories:
- $$
  \hat g = \frac{1}{|\mathcal{D}|} \sum_{\tau\in\mathcal{D}} \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, R(\tau)
  $$
- No model of the environment needed: just log-probabilities and returns.

</v-clicks>

<!--
Each epoch collects 10 trajectories with the current policy (green: reached the goal, red: lava, grey: timed out), then takes one gradient step. Every action in a high-return trajectory is pushed up, every action in a low-return one pushed down. This is exactly what your notebook does. Use + to speed it up.
-->

---
layout: code
kicker: Part 3
---

# Implementing it

::left::

<v-clicks>

- The policy: a network outputs logits; `Categorical` turns them into $\pi_\theta(\cdot \mid s)$.
- The "loss": its gradient is exactly $\hat g$. It needs $\log \pi_\theta(a \mid s)$ (`log_prob`):
- $$
  \begin{array}{ll}
  \text{categorical:} & \log \operatorname{softmax}(\text{logits})_a \\
  \text{Gaussian:} & -\tfrac12 \sum_i \Big[\frac{(a_i-\mu_i)^2}{\sigma_i^2} + 2\log\sigma_i + \log 2\pi\Big]
  \end{array}
  $$
- The weights: every step of an episode gets that episode's return $R(\tau)$.
- One epoch: collect a batch with the current policy, then take one gradient step.
- Not a real loss: it doesn't measure performance. Track average return instead.

</v-clicks>

::right::

```python {all|1-8|10-12|10-12|14-15|17-21|10-12}{at:0}
logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)

def get_action(obs):
    return get_policy(obs).sample().item()

def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()

# after each episode, in train_one_epoch()
batch_weights += [ep_ret] * ep_len

# then one gradient step per epoch
optimizer.zero_grad()
batch_loss = compute_loss(obs=..., act=..., weights=...)
batch_loss.backward()
optimizer.step()
```

<!--
This is the code in 1_policy_grad.ipynb (Spinning Up's 1_simple_pg.py). Each bullet highlights the matching lines. log_prob is the only policy-specific piece: for a categorical policy it is the log-softmax of the logits; for a diagonal Gaussian it is the closed form shown. The minus sign is there because optimizers minimize. Its data depends on θ, and the "loss" can go down while performance gets worse, so never judge training by it.
-->

---
layout: demo
scene: eglp
kicker: Part 3
---

# The expected grad-log-prob lemma

::left::

<v-clicks>

- $$
  \mathbb{E}_{x\sim P_\theta}\big[\nabla_\theta \log P_\theta(x)\big] = 0
  $$
- Proof: differentiate $\int P_\theta(x) = 1$, then apply the log-derivative trick.
- For a softmax policy: $\nabla_\theta \log \pi(a \mid s) = \mathbf{e}_a - \pi(\cdot \mid s)$
- Each sample is a big, noisy vector. Averaged over many samples, it goes to zero.
- This lemma lets us drop and add terms in the gradient, coming next.

</v-clicks>

<!--
We sample actions from π(·|start) and plot ∇ log π for the latest sample (top) and the running average (bottom), on the same scale. Single samples are large; the average shrinks toward 0 roughly like 1/√N. This zero-mean property is what justifies reward-to-go and baselines.
-->

---
layout: demo
scene: reward-to-go
kicker: Part 3
---

# Don't let the past distract you

::left::

<v-clicks>

- With $R(\tau)$, each action is credited with rewards earned *before* it was taken.
- Reward-to-go: $\hat R_t = \sum_{t'=t}^{T} r_{t'}$
- $$
  \nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \hat R_t\Big]
  $$
- Past rewards add zero on average but plenty of noise. Dropping them means fewer episodes are needed.

</v-clicks>

<!--
After each episode, the pills show the weight given to each action. Toggle between R(τ), where every action gets the same number, and reward-to-go, where an action only gets credit for what came after it. In code: batch_weights += list(reward_to_go(ep_rews)).
-->

---
layout: demo
scene: baselines
kicker: Part 3
---

# Baselines

::left::

<v-clicks>

- For any function of state alone: $\mathbb{E}_{a\sim\pi}\big[\nabla_\theta \log \pi_\theta(a \mid s)\, b(s)\big] = 0$
- $$
  \nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\,\big(\hat R_t - b(s_t)\big)\Big]
  $$
- The usual choice: $b(s) = V^\pi(s)$, learned by a network $V_\phi$ with a squared-error loss.
- $\hat R_t - V(s_t)$ estimates the advantage from Part 1: was this action better than usual from here?

</v-clicks>

<!--
The shading is V^π for this policy (computed exactly here; in practice V_φ is learned by regressing onto R̂_t). Now the pills are R̂_t − V(s_t): green actions turned out better than expected and get reinforced, red ones worse. Compare with the previous slide, where nearly everything had the same sign.
-->

---
layout: demo
scene: forms
kicker: Part 3
---

# Other forms of the policy gradient

::left::

<v-clicks>

- $$
  \nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \Phi_t\Big]
  $$
- $\Phi_t$ can be: $R(\tau)$ · $\hat R_t$ · $\hat R_t - b(s_t)$ · $Q^\pi(s_t,a_t)$ · $A^\pi(s_t,a_t)$
- All give the same expected gradient. They differ in variance, which sets how fast you learn.
- Using the advantage $A^\pi$ gives Vanilla Policy Gradient (VPG). GAE is a good way to estimate it.

</v-clicks>

<!--
Three learners, three seeds each, same learning rate, 10 episodes per epoch; only Φ differs. R(τ) is slowest; reward-to-go removes the noise of past rewards; the learned baseline removes more. Early on the baseline is still being learned, so it can briefly trail plain reward-to-go.
-->

---
layout: demo
scene: closing
kicker: What's next
---

# Now: the notebook

::left::

<v-clicks>

- Same algorithm, bigger world: CartPole's observation is 4 numbers, with 2 actions.
- The table of logits becomes a small neural network: `logits_net`.
- Start with weights = $R(\tau)$, then switch to reward-to-go and compare the learning curves.
- Next in Spinning Up: VPG, TRPO and PPO.

</v-clicks>

<!--
Hand off to 1_policy_grad.ipynb. Exercise: replace batch_weights += [ep_ret] * ep_len with reward-to-go and compare. The robot here follows the optimal policy.
-->
