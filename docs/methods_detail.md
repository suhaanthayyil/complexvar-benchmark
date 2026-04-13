# Methods Detail

## Interface definition

A residue is defined as interface-associated if its minimum heavy-atom distance
to the partner chain is at most `10.0 A`.

## Graph edges

- sequential residue adjacency within a chain
- spatial residue-residue edge when minimum heavy-atom distance is at most
  `8.0 A`
- inter-chain flag on each edge

## Variant-centered graph extraction

- cache full graphs once
- extract local subgraphs within `12.0 A` of the mutated residue
- deterministic node cap when needed

## Evaluation

- primary analysis on interface variants
- secondary analysis on non-interface variants
- sample-level plus macro-averaged grouped summaries
