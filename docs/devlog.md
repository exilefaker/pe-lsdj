# Parallel Experiments LSDJ 2.0 - developer's log

## Milestones

- [2015] Initial idea/pass at PELSDJ
- [7/3/2020] exileFaker "Parallel Experiments LSDJ" EP release
- [2/12/2026] Initial commit of v2
- [2/14❤️/2026] Tokenization pipeline working 
- [2/18/2026] First embedding pipeline working 
- [2/20/2026] First model + train loop
- [2/23/2026] Model v2
- [2/26/2026] Model v3-v4
- [2/27/2026] Model v5
- [3/3/2026] First generation pipeline

## Model version history

| Version | Class           | Description                                                  | Works |
|---------|-----------------|--------------------------------------------------------------|-------|
|   v1    | LSDJTransformer | cross-entropy loss used for all outputs including entity IDs | ✅    |
|   v2    | LSDJTransformer | "entity loss" (CE on entity features) used in place of instr, table, groove IDs | ✅ |
|   v3    | LSDJTransformer | hierarchical entity un-embedding covering softsynths and inner tables; MSE loss for continuous values | ✅ |
|   v4    | LSDJTransformer | true hierarchical decoding mirroring embedding structure. Predict individual groove/table steps rather than a single value | ❌ (OOM) |
|   v5    | LSDJTransformer | Conditionally generate FX values | ✅ |
|   v6    | LSDJTransformer | Add constrative loss for entity embedder/decoder | ❌ (OOM) |

