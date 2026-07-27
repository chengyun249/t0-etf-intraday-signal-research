# Changelog

## 2.0.0 — 2026-07-27

- Consolidated the publishable T+0 project at the repository root.
- Archived mixed ETF-rotation and legacy-2025 prototypes locally.
- Removed the duplicate 560MB nested zip.
- Added lagged-VWAP, gap-aware and adverse-slippage stop execution.
- Added a point-in-time daily dynamic universe.
- Added purged date-block validation without numeric ETF-code features.
- Unified cost conventions and bounded dependency versions.
- Added data-independent tests and rewrote documentation.
- Renamed the range/noise proxy from expected edge to movement budget.
- Replaced profit-factor-rewarding selection with trade-date-clustered lower-confidence-bound scoring.
- Raised sample gates, changed walk-forward updates to weekly, and made failed development gates explicit.
- Added bootstrap-based strategy acceptance reports; current fixed-combination candidate is research-only.
