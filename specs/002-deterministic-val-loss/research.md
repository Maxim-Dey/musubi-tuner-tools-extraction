# Research decisions

Independent data/config, math/evaluation and checkpoint agents inspected the actual retained code. Decisions are fixed in [plan.md](plan.md).

| Decision | Evidence/rationale | Rejected alternative |
|---|---|---|
| Reuse compute_loss with fixed sigma | Existing weighted MSE/reduction is already shared; scheduler lookup otherwise shifts sigma | A similar independent val loss changes the contract |
| Strict source-linked records | Current cache enumeration skips missing text cache and can include stale caches | Ordinary val loader silently changes population |
| Remove role field | Existing schema lacks role; val_dataset_config identifies purpose | Unnecessary new schema roles |
| Deterministic encoding | Existing VAE posterior.mode and center crop | Random re-encoding each evaluation |
| Dedicated CPU FP32 noise | Precise portable byte/seed contract within environment | Mutating global training RNG |
| Main-only unwrapped evaluation | Existing trainer manually reduces gradients; unwrap avoids DDP forward communication | Padded distributed sampler duplicates images |
| Persist explicit step/run and verify events | Loop resets global_step, accelerator setup creates timestamped runs | Infer completed steps from Accelerator.step/filename |
| Canonical state hooks/all-rank save | Separate exports duplicate adapter; main-only state loses other rank RNG | Two weight copies or fabricated RNG inventories |

No open design clarification remains. Actual server/model/independent-val availability is checked in E, never assumed.
