# Model version history

| Version | Class           | Description                                                  | Works |
|---------|-----------------|--------------------------------------------------------------|-------|
|   v1    | LSDJTransformer | cross-entropy loss used for all outputs including entity IDs | ✅    |
|   v2    | LSDJTransformer | "entity loss" (CE on entity features) used in place of instr, table, groove IDs | ✅ |
|   v3    | LSDJTransformer | hierarchical entity un-embedding covering softsynths and inner tables; MSE loss for continuous values | ✅ |
|   v4    | LSDJTransformer | true hierarchical decoding mirroring embedding structure. Predict individual groove/table steps rather than a single value | ❌ |
|   v5    | LSDJTransformer | Conditionally generate FX values | ✅ |
