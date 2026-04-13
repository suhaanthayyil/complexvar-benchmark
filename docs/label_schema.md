# Label Schema

## IntAct / IMEx binary task

- Positive:
  - `disrupting`
  - `decreasing`
- Negative:
  - `no effect`
- Excluded:
  - `increasing`
  - `causing`
  - `undefined`
  - conflicted or unmapped labels

## SKEMPI tasks

### Regression

- Target: reported `ddG`

### Binary

- Positive if `ddG >= 1.5 kcal/mol`
- Negative if `|ddG| <= 0.5 kcal/mol`
- Ambiguous middle band excluded from the primary binary task

## ClinVar external disease-variant set

- Positive:
  - `Pathogenic`
  - `Likely pathogenic`
  - `Pathogenic/Likely pathogenic`
- Negative:
  - `Benign`
  - `Likely benign`
  - `Benign/Likely benign`
- Excluded:
  - VUS
  - conflicting interpretations
  - somatic-only categories
  - risk factor
  - association
  - drug response
