"""
Example usage of Speculative Decoding Draft Verifier Skill.
"""

from client import SpeculativeDecoder


def main():
    print("=== Speculative Decoding Draft Verifier Demonstration ===")
    decoder = SpeculativeDecoder(gamma=4)

    # Step 1: Draft model generates 4 speculative tokens
    draft_tokens = ["def", "calculate", "_", "fibonacci"]
    draft_probs = [0.92, 0.85, 0.70, 0.65]
    target_probs = [0.95, 0.88, 0.72, 0.60]
    random_draws = [0.20, 0.45, 0.60, 0.85]  # Token 4 rejected

    res1 = decoder.verify_tokens(draft_tokens, draft_probs, target_probs, random_draws)
    print("Step 1 Results:")
    print("  Draft proposed:   ", draft_tokens)
    print("  Accepted tokens:  ", res1["accepted_tokens"])
    print("  Rejected at index:", res1["rejected_at_index"])
    print("  Tokens emitted:   ", res1["effective_tokens_generated"])

    # Step 2: Perfect match
    draft2 = ["(", "n", ":", "int"]
    draft_p2 = [0.98, 0.95, 0.90, 0.85]
    target_p2 = [0.99, 0.97, 0.92, 0.88]
    randoms2 = [0.10, 0.15, 0.20, 0.30]

    res2 = decoder.verify_tokens(draft2, draft_p2, target_p2, randoms2)
    print("\nStep 2 Results:")
    print("  Draft proposed:   ", draft2)
    print("  Accepted tokens:  ", res2["accepted_tokens"])
    print("  Tokens emitted:   ", res2["effective_tokens_generated"])

    stats = decoder.get_aggregate_stats()
    print("\n--- Aggregate Decoding Metrics ---")
    print("Mean Acceptance Rate: ", f"{stats['mean_acceptance_rate'] * 100:.1f}%")
    print("Estimated Speedup:    ", f"{stats['estimated_speedup_factor']}x")


if __name__ == "__main__":
    main()
