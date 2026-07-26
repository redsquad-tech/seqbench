# Third-party datasets

`seqbench` contains download and normalization scripts, not third-party
dataset rows. Users obtain each dataset from its original upstream location
and remain responsible for its terms.

| Dataset | Upstream | Notes |
|---|---|---|
| MRCR | `openai/mrcr` | MIT dataset card |
| bAbI | ParlAI bAbI archive | BSD |
| BABILong | `RMT-team/babilong` | Apache-2.0 code/data components; includes bAbI |
| CLUTRR | `CLUTRR/v1` | CC-BY-NC-4.0; non-commercial restriction |
| ProofWriter | `tasksource/proofwriter` | Mirror does not declare a clear data license |
| ReCOGS | `frankaging/ReCOGS` | MIT |
| SLOG | `bingzhilee/SLOG` | MIT; generalization ZIP is intentionally password-protected against contamination |

The source revisions used by the scripts are listed in
`tools/datasets_manifest.json`. The Apache-2.0 license of `seqbench` covers
only this repository's own code and specifications.

