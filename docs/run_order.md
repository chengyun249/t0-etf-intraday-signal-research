# Run Order

This project is a research pipeline for A-share T+0 ETF 1-minute data. The canonical code directory is `scripts/`.

## Full Pipeline

1. Download 1-minute ETF data and build the candidate universe.

```bash
python scripts/download_t0_etf_1min_2022_2024.py
```

2. Build the processed intraday feature panel.

```bash
python scripts/build_t0_intraday_bar_panel_2022_2024.py
```

3. Audit panel quality and backtest assumptions.

```bash
python scripts/audit_t0_intraday_panel_and_backtest_assumptions.py
```

4. Run the adaptive intraday momentum backtest.

```bash
python scripts/backtest_adaptive_intraday_momentum.py
```

5. Run the combined ORB / noise / relative-value strategy.

```bash
python scripts/backtest_combined_orb_noise_rv.py
```

6. Run daily walk-forward validation.

```bash
python scripts/backtest_combo_walk_forward_daily.py
```

7. Diagnose raw forward returns after combo signals.

```bash
python scripts/diagnose_combo_signal_forward_returns.py
```

8. Train the LightGBM direction model.

```bash
python scripts/train_intraday_lgbm_direction_model.py
```

9. Train the LightGBM return/downside-risk model.

```bash
python scripts/train_intraday_lgbm_return_model.py
```

## Notes

- Raw minute data and full processed parquet panels are intentionally excluded from the public repository.
- Existing summary reports and tables are stored under `reports/`.
- Most scripts accept `--panel-file`, `--out-dir`, date range, cost, and scope parameters for partial reruns.
