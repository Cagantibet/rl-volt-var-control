"""Five-point Volt/VAR and Volt/Watt curves.

A curve is a ``(2, 5)`` array: row 0 holds the voltage breakpoints (pu, non-decreasing),
row 1 the output at each breakpoint. Outside the breakpoints the output is held constant,
as IEEE 1547 inverters do.

* Volt/VAR output: reactive power in pu of inverter kVA (+ = inject, raises voltage).
* Volt/Watt output: active-power ceiling in pu of available power.
"""

from __future__ import annotations

import numpy as np

from .config import CurveConfig

N_POINTS = 5
PARAMS_PER_PV = 4 * N_POINTS  # VV volts, VV Q, VW volts, VW P


def evaluate(curves: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate one curve per inverter: ``curves`` is (n, 2, 5), ``v`` is (n,)."""
    return np.array([np.interp(vi, c[0], c[1]) for c, vi in zip(curves, v)])


def _scale(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + (a + 1.0) / 2.0 * (hi - lo)


def action_to_curves(action: np.ndarray, n_pv: int, cfg: CurveConfig,
                     q_max_pu: float) -> tuple[np.ndarray, np.ndarray]:
    """Map a flat policy action in [-1, 1] to physically valid curves for every inverter.

    Out-of-range actions are clipped. Breakpoint voltages are sorted ascending. Reactive
    power is sorted descending (inject at low V, absorb at high V) and active power is
    sorted descending (curtail more as V rises), so every curve has a standard shape.
    """
    a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0).reshape(n_pv, PARAMS_PER_PV)
    k = N_POINTS
    vv_v = np.sort(_scale(a[:, 0:k], *cfg.vv_voltage_range), axis=1)
    vv_q = -np.sort(-(a[:, k:2 * k] * q_max_pu), axis=1)
    vw_v = np.sort(_scale(a[:, 2 * k:3 * k], *cfg.vw_voltage_range), axis=1)
    vw_p = -np.sort(-_scale(a[:, 3 * k:4 * k], cfg.vw_p_min, 1.0), axis=1)
    return np.stack([vv_v, vv_q], axis=1), np.stack([vw_v, vw_p], axis=1)


# ---------------------------------------------------------------- reference curves
def _tile(curve, n_pv: int) -> np.ndarray:
    return np.tile(np.asarray(curve, dtype=float)[None], (n_pv, 1, 1))


def vv_disabled(n_pv: int) -> np.ndarray:
    return _tile([[0.90, 0.95, 1.00, 1.05, 1.10], [0, 0, 0, 0, 0]], n_pv)


def vw_disabled(n_pv: int) -> np.ndarray:
    return _tile([[1.00, 1.025, 1.05, 1.075, 1.10], [1, 1, 1, 1, 1]], n_pv)


def vv_ieee1547(n_pv: int, q_max_pu: float = 0.44) -> np.ndarray:
    """IEEE 1547-2018 Category B default Volt/VAR curve (V_ref = 1.0 pu)."""
    q = q_max_pu
    return _tile([[0.92, 0.98, 1.00, 1.02, 1.08], [q, 0, 0, 0, -q]], n_pv)


def vw_ieee1547(n_pv: int) -> np.ndarray:
    """IEEE 1547-2018 default Volt/Watt curve: full power to 1.06 pu, 20 % at 1.10 pu."""
    return _tile([[1.00, 1.03, 1.06, 1.08, 1.10], [1, 1, 1, 0.6, 0.2]], n_pv)


def vv_aggressive(n_pv: int, q_max_pu: float = 0.44) -> np.ndarray:
    """Tighter hand-tuned Volt/VAR curve: full reactive range over 0.95-1.05 pu."""
    q = q_max_pu
    return _tile([[0.95, 0.98, 1.00, 1.02, 1.05], [q, q / 2, 0, -q / 2, -q]], n_pv)


def vw_aggressive(n_pv: int) -> np.ndarray:
    """Tighter hand-tuned Volt/Watt curve: curtail from 0.98 pu to zero at 1.05 pu."""
    return _tile([[0.95, 0.98, 1.00, 1.02, 1.05], [1, 1, 0.8, 0.4, 0]], n_pv)
