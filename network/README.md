# Network Module

Stage 6 provides the versioned `hexadeca-v1-16p` feature encoder, configurable
residual policy-value model, legal-action policy masking, and composite
AlphaZero objective. The model emits raw logits for all 256 actions; callers
apply legal masking before loss or search normalization.

The value head predicts normalized Black score, normalized White score, and
the current player's win probability. Raw score targets are divided by the
`score_normalizer` recorded in `NetworkSpecification` before entering this
module. Optimizers, training loops, MCTS, and checkpoint lifecycle management
remain reserved for their later stages.
