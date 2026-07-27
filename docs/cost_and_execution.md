# Cost and execution conventions

The canonical v2 convention is a 2bp one-way fixed cost, with sensitivity at 0/1/2/3/5/10bp. A round trip therefore subtracts twice the selected fixed cost. Stop fills additionally use 1bp adverse execution slippage by default.

These values are scenario inputs, not measured spreads. A real feasibility study needs bid/ask quotes, depth, order size, latency, rejected orders and impact. Any result that disappears between 2bp and 5bp must be described as execution-sensitive.
