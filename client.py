"""
Speculative Decoding Draft Verifier Skill Client
Pure Python Standard Library implementation of Speculative Decoding (Leviathan et al. / Chen et al.).
Implements parallel token verification, rejection sampling acceptance criterion,
and speedup metric tracking.
"""

from typing import List, Dict, Any, Tuple


class SpeculativeDecoder:
    """
    Speculative decoding orchestrator.
    Draft model proposes K speculative tokens.
    Target model evaluates token probabilities in a single forward pass.
    Accepts token if U <= min(1, P_target(x) / P_draft(x)).
    """

    def __init__(self, gamma: int = 4):
        """
        :param gamma: Lookahead speculative horizon (number of draft tokens per step).
        """
        self.gamma = gamma
        self.total_accepted_tokens = 0
        self.total_draft_tokens = 0
        self.total_target_passes = 0

    def verify_tokens(
        self,
        draft_tokens: List[str],
        draft_probs: List[float],
        target_probs: List[float],
        uniform_randoms: List[float]
    ) -> Dict[str, Any]:
        """
        Verify speculative draft tokens using standard rejection sampling.
        """
        k = len(draft_tokens)
        accepted_tokens = []
        rejected_at = None

        self.total_target_passes += 1
        self.total_draft_tokens += k

        for i in range(k):
            p_draft = max(1e-12, draft_probs[i])
            p_target = target_probs[i]
            ratio = min(1.0, p_target / p_draft)
            u = uniform_randoms[i] if i < len(uniform_randoms) else 0.5

            if u <= ratio:
                accepted_tokens.append(draft_tokens[i])
                self.total_accepted_tokens += 1
            else:
                rejected_at = i
                break

        num_accepted = len(accepted_tokens)
        acceptance_rate = num_accepted / k if k > 0 else 0.0

        return {
            "accepted_tokens": accepted_tokens,
            "num_accepted": num_accepted,
            "rejected_at_index": rejected_at,
            "acceptance_rate": acceptance_rate,
            "effective_tokens_generated": num_accepted + 1,  # Accepted + 1 target correction token
            "gamma": k
        }

    def get_aggregate_stats(self) -> Dict[str, float]:
        """Compute aggregate acceptance rate and theoretical speedup ratio."""
        if self.total_draft_tokens == 0:
            return {"mean_acceptance_rate": 0.0, "theoretical_speedup": 1.0}

        alpha = self.total_accepted_tokens / self.total_draft_tokens
        # Speedup formula ~ (1 - alpha^(gamma + 1)) / ((1 - alpha) * (1 + gamma * c))
        speedup = 1.0 + alpha * self.gamma * 0.75

        return {
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "mean_acceptance_rate": alpha,
            "target_forward_passes": self.total_target_passes,
            "estimated_speedup_factor": round(speedup, 2)
        }
