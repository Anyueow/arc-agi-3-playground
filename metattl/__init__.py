"""Meta-TTL for ARC-AGI-3: learning an adaptation policy (Lou et al., arXiv 2604.00830).

Two policies, two loops:

  ACTOR pi(. | rho)      the LLM brain (agents/__init__.py). Its system prompt
                         contains a "guide" rho that is the only thing that
                         changes between episodes.
  META-AGENT f_phi       an LLM governed by the meta-prompt phi. After each
                         episode it reads the history and rewrites rho.

  INNER LOOP (session.py, Algorithm 1 "RunTTL"): one game, K episodes.
      episode k = one attempt at the current level (ends on death, level
      cleared, or the action budget). After a failed attempt the level
      restarts; after a clear the next episode starts on the next level.
      Score: W-AUC = sum_k k*J_k / sum_k k*J_max  (later episodes weigh more).

  OUTER LOOP (evolve.py, Algorithm 2): evolutionary search over phi.
      sample parent phi from the expert pool + a training game
      -> run session -> proposer LLM reflects and proposes a new phi
      -> local validation (same game, must beat parent)
      -> global validation on every validation game; becomes the expert
         for each game it sets a new best on
      -> finally pick phi* by mean per-game z-score.

Files:
  episode.py     RunActor: one episode, its return J and a text summary
  session.py     RunTTL + W-AUC
  meta_agent.py  Adapt(phi, rho, history) -> new rho
  evolve.py      Propose, expert pool, Algorithm 2, z-score selection
  prompts.py     seed meta-prompt phi0 and the proposer's instructions
"""
