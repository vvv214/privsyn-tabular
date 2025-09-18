# Privacy Budget Notes

PrivSyn supports two synthesis engines—PrivSyn (rho-CDP) and AIM (approximate zCDP). This note summarises how epsilon, delta, and rho play together so you can rationalise the budget before deploying to production.

## 1. PrivSyn

- **Interface**: the API accepts an `epsilon`/`delta` pair. Internally the PrivSyn library converts these to a rho-zCDP budget using `method/util/rho_cdp.py`.
- **Iterations**: the number of update iterations (`update_iterations`) and consistency passes (`consist_iterations`) consume the same global budget. Reducing them lowers overall runtime and the amount of per-marginal noise but also reduces fidelity.
- **Preprocessing**: discretisation (e.g. PrivTree binning) can optionally draw a portion of the budget via `dp_budget_fraction` in the UI. When set, the fraction reduces the budget available to the main synthesiser.

### Practical Tips
- Prefer providing tight numeric bounds to avoid the extra noise introduced by clipping.
- For categorical columns, trimming rare values (or mapping them to the special token) improves stability.
- When exploring, start with ε between 0.5 and 2.0 and δ ≤ 1e-5; adjust upward only if the quality is insufficient and privacy policy allows it.

## 2. AIM

- AIM consumes the provided `(epsilon, delta)` pair directly.
- Runtime is sensitive to the number of iterations and workload size; for quick evaluations consider reducing the workload prior to deployment or sampling a subset of columns.

## 3. Rho Conversion

The helper `method/util/rho_cdp.py` implements conversions between (ε, δ) and ρ (zCDP). Keep in mind that when composing multiple DP mechanisms (e.g., preprocessing + synthesis + evaluation), the rho values add linearly:

```
ρ_total = ρ_preprocess + ρ_synthesis + ρ_evaluation
```

Use the conversion utilities if you need to enforce a global privacy ledger across multiple pipelines.

## 4. Evaluation Considerations

The `/evaluate` endpoint uses histogram-aware TVD which is **not** differentially private—it's intended only for offline quality assessment. Avoid exposing it to untrusted users or run it only on subsampled/aggregated outputs.

## 5. Future Work

- Pluggable privacy budget managers (e.g., Google DP accounting) to track consumption across multiple runs.
- UI hints for typical epsilon/delta values per domain (health, finance, etc.).

For more background on CDP and the PrivSyn method, see the references in the `pdf/` folder.
