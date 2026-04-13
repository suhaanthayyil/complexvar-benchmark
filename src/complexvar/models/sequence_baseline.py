"""Compatibility wrapper for the sequence-only baseline."""

from complexvar.models.sequence import (  # noqa: F401
    ESM2_MODEL_NAME,
    SequenceEncoding,
    SequenceMLP,
    build_sequence_feature_vector,
    esm2_mutation_embedding,
    simple_sequence_window_embedding,
)
