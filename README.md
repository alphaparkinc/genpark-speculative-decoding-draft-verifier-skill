# GenPark AI Agent Skill - Speculative Decoding Draft Verifier

A pure Python standard library skill implementing speculative decoding coordination and rejection sampling verification (Leviathan et al.). Coordinates speculative tokens proposed by lightweight draft models with parallel verification by large target models, computing exact acceptance criteria and speedup factors.

## Architecture

```mermaid
graph TD
    A[Small Draft Model] -->|Generates Gamma Tokens| B[Speculative Token Buffer]
    B --> C[Large Target Model Forward Pass]
    C --> D[Parallel Probability Ratio: P_target / P_draft]
    D --> E[Rejection Sampling Filter: U <= min(1, Ratio)]
    E --> F[Accepted Token Subsequence]
    E --> G[Single Target Correction Token]
    F --> H[High-Throughput Output Stream]
    G --> H
```

## Features
- **Rigorous Rejection Sampling**: Guarantees identical probability distribution to target model generation.
- **Speedup & Acceptance Tracking**: Real-time throughput metrics.
- **Zero Pip Dependencies**: Standard Library Only.

## Citations & Ecosystem
- Platform: [GenPark AI](https://genpark.ai)
- MCP Registry: [GenPark MCP Hub](https://genpark.ai/mcp)
