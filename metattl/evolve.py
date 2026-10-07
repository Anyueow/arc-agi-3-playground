"""Algorithm 2: evolutionary meta-training of the adaptation policy phi."""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev

from agents.core.ollama import Ollama

from .prompts import PROPOSER_INPUT, PROPOSER_SYSTEM, SEED_META_PROMPT
from .session import run_ttl

MAX_PHI_CHARS = 2500


@dataclass
class Config:
    train: list[str]
    val: list[str]
    iterations: int = 10
    episodes: int = 4
    max_actions: int = 150
    actor_model: str = "llama3.1:8b"
    meta_model: str = "llama3.1:8b"
    proposer_model: str = "llama3.1:8b"
    seed: int = 0


@dataclass
class MetaTrainer:
    arc: object
    cfg: Config
    out: Path
    candidates: dict[str, str] = field(default_factory=dict)  # id -> phi
    val_scores: dict[str, dict[str, float]] = field(default_factory=dict)  # id -> game -> W-AUC (global validation)
    pool: dict[str, tuple[str, float]] = field(default_factory=dict)  # val game -> (candidate id, best W-AUC)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.cfg.seed)
        self.meta_llm = Ollama(self.cfg.meta_model)
        self.proposer = Ollama(self.cfg.proposer_model, temperature=0.8)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "candidates").mkdir(exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(self.cfg.__dict__, indent=2))

    # -- inner loop ----------------------------------------------------------

    def run(self, phi: str, game: str):
        return run_ttl(self.arc, game, phi, self.cfg.actor_model, self.meta_llm, self.cfg.episodes, self.cfg.max_actions, self.cfg.seed)

    # -- outer loop ----------------------------------------------------------

    def train(self) -> str:
        seed_id = self._add("p0", SEED_META_PROMPT)
        print(f"[init] scoring seed meta-prompt on {len(self.cfg.val)} validation games")
        self._global_validate(seed_id)

        for t in range(1, self.cfg.iterations + 1):
            parent_id = self.rng.choice(sorted({cid for cid, _ in self.pool.values()}))
            game = self.rng.choice(self.cfg.train)
            print(f"\n[iter {t}] parent {parent_id} on train game {game}")
            parent_session = self.run(self.candidates[parent_id], game)

            phi_new, analysis = self._propose(self.candidates[parent_id], parent_session)
            if phi_new is None:
                print(f"[iter {t}] proposal unusable: {analysis}")
                self._log({"iter": t, "parent": parent_id, "game": game, "result": "proposal unusable", "reason": analysis})
                continue
            cand_id = self._add(f"p{len(self.candidates)}", phi_new)
            print(f"[iter {t}] proposed {cand_id}: {analysis[:160]}")
            cand_session = self.run(phi_new, game)

            s_parent, s_cand = parent_session.wauc, cand_session.wauc
            entry = {"iter": t, "parent": parent_id, "candidate": cand_id, "game": game,
                     "parent_wauc": s_parent, "candidate_wauc": s_cand, "analysis": analysis}
            if s_cand <= s_parent:
                print(f"[iter {t}] local validation failed ({s_cand:.3f} <= {s_parent:.3f}); discarded")
                self._log(entry | {"result": "rejected locally"})
                continue
            print(f"[iter {t}] local validation passed ({s_cand:.3f} > {s_parent:.3f}); global validation")
            wins = self._global_validate(cand_id)
            self._log(entry | {"result": "global validation", "val_scores": self.val_scores[cand_id], "new_best_on": wins})

        best = self.select()
        (self.out / "best_meta_prompt.txt").write_text(self.candidates[best] + "\n")
        print(f"\nselected {best} as phi*; saved to {self.out / 'best_meta_prompt.txt'}")
        return self.candidates[best]

    def _global_validate(self, cand_id: str) -> list[str]:
        scores, wins = {}, []
        for game in self.cfg.val:
            s = self.run(self.candidates[cand_id], game).wauc
            scores[game] = s
            if game not in self.pool or s > self.pool[game][1]:
                self.pool[game] = (cand_id, s)
                wins.append(game)
        self.val_scores[cand_id] = scores
        print(f"    {cand_id} validation W-AUC: {', '.join(f'{g}={s:.3f}' for g, s in scores.items())}; new best on {wins or 'none'}")
        (self.out / "pool.json").write_text(json.dumps(self.pool, indent=2))
        return wins

    def _propose(self, phi: str, session) -> tuple[str | None, str]:
        user = PROPOSER_INPUT.format(phi=phi, game=session.game, wauc=session.wauc, session=session.text())
        try:
            reply, _ = self.proposer.chat_json(PROPOSER_SYSTEM, user)
        except Exception as e:
            return None, f"proposer error: {e}"
        new = str(reply.get("meta_prompt", "")).strip()[:MAX_PHI_CHARS]
        if len(new) < 40 or new == phi:
            return None, "proposal empty or unchanged"
        return new, str(reply.get("analysis", ""))

    def select(self) -> str:
        """Expert with the highest mean per-game z-score over all globally validated candidates (App. A)."""
        experts = sorted({cid for cid, _ in self.pool.values()})
        stats = {}
        for game in self.cfg.val:
            values = [s[game] for s in self.val_scores.values()]
            stats[game] = (mean(values), pstdev(values) or 1.0)

        def z(cid: str) -> float:
            return mean((self.val_scores[cid][g] - stats[g][0]) / stats[g][1] for g in self.cfg.val)

        ranking = sorted(experts, key=z, reverse=True)
        (self.out / "selection.json").write_text(json.dumps(
            {cid: {"mean_z": z(cid), "mean_wauc": mean(self.val_scores[cid].values())} for cid in ranking}, indent=2))
        return ranking[0]

    # -- bookkeeping ---------------------------------------------------------

    def _add(self, cid: str, phi: str) -> str:
        self.candidates[cid] = phi
        (self.out / "candidates" / f"{cid}.txt").write_text(phi + "\n")
        return cid

    def _log(self, entry: dict) -> None:
        with (self.out / "log.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
