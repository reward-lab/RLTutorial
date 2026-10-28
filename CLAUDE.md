# RLTutorial

Lecture material following OpenAI Spinning Up.

## Files
- `1_policy_grad.ipynb` — simplest policy gradient on CartPole (torch + gymnasium).
- `rl_intro_gridworld.html` — self-contained animated gridworld for the "Key Concepts in RL" lecture
  (https://spinningup.openai.com/en/latest/spinningup/rl_intro.html). Open in a browser; also published at
  https://claude.ai/artifact/TnXGyKVWBHxaHTvTLVwvVw.

## Gridworld lecture page
- 5×5 grid, start bottom-left, goal +10 (top-right), lava −50 (changed from −10 so policy gradient doesn't learn to suicide into lava),
  step −1, walls; "slip" = prob of sliding sideways. Constants R_GOAL/R_LAVA/R_STEP.
- 16:9 slide deck (1600×900 stage scaled to fit), 27 slides; footer shows current part. Covers Spinning Up Parts 1 and 3:
  1 Title · 2 Part 1 header · 3 What can RL do · 4 Meet the gridworld · 5 Loop · 6 States/obs · 7 Action spaces · 8 Policies ·
  9 Stochastic policies in practice · 10 Trajectories · 11 Reward & return · 12 RL problem J(π) · 13 V/Q · 14 V*/Q* ·
  15 Bellman · 16 Advantage · 17 Formal picture · 18 Part 3 header · 19 Gradient ascent on the policy · 20 Derivation: ingredients ·
  21 Derivation: result + batch estimate ĝ · 22 Implementing it (code, lines highlight per bullet) · 23 EGLP lemma ·
  24 Reward-to-go · 25 Baselines · 26 Other forms Φ_t (race: R(τ) vs reward-to-go vs baseline) · 27 Now: the notebook.
- Part 3 uses finite-horizon undiscounted returns, T3=50, tabular softmax PG (makePG3: kinds 'R','rtg','base', α=0.01, batch 10).
- Part 2 (Kinds of RL Algorithms, 10 slides) is still in the code but hidden: set SHOW_PART2=true to restore it.
  Its learning-curve chart range [−12,0] was tuned for lava −10 and would need adjusting.
- Keys: →/space/PageDown next build or slide, ←/PageUp back, p play/pause, . step, r reset, +/- speed, n speaker notes, f fullscreen. `#s5` jumps to slide 5.
- Randomness is seeded (SEED=42, mulberry32 replacing Math.random); reseeded per slide on entry and on reset (r),
  so runs replay identically. Change SEED at the top of the script for a different run.
- Slide 6 (States/obs) simplified: no 3×3-patch/partial-observation toggle; states o = s (fully observed).
  (The 3×3 patch never aliased in this layout anyway; a plus-shaped view would alias (3,1)/(3,3).)
- Slide 9 shows sampling only (softmax; a = μ + σ⊙z), no log π. The log π formulas (categorical, Gaussian) live on slide 22
  as a build under the "loss" bullet (HL=[1,2,2,3,4,2]).
- Not yet committed.
- (Part 2, hidden) bullet reads "Policy gradients raise π(a|s) for actions with positive advantage."
- Known quirks (offered fixes, not yet requested): slide 9 histogram fixed to [−20, 10] and rollouts capped at 200 steps;
  slide 11 Q* argmax circles show at sweep 0; temperature slider absent on slide 9; with slip 0.6 + γ 0.99 the optimal
  policy walks into lava (lava −10 ≈ max discounted step cost) — a reward-design talking point.
