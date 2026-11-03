"""Thin wrapper around OpenDSS (via opendssdirect.py) for the IEEE 123-node feeder with PV.

The circuit is compiled once. Scenarios are switched by overwriting the load-shape and
irradiance-shape multipliers in place, so no files are written and nothing is recompiled.
"""

from __future__ import annotations

import os

import numpy as np
import opendssdirect as dss

from .config import Config
from .profiles import HOURS, DayProfiles

LOAD_SHAPE = "load_daily"


class Feeder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.sites = cfg.pv.site_list()
        self.pv_names = [name for name, _, _ in self.sites]
        self.n_pv = len(self.sites)
        self.kva = cfg.pv.kva
        self.pmpp = cfg.pv.pmpp
        self._irradiance = np.zeros((self.n_pv, HOURS))
        self.hour: int | None = None
        self._compile()

    # ------------------------------------------------------------------ setup
    def _compile(self) -> None:
        dss.Basic.ClearAll()
        cwd = os.getcwd()
        try:
            dss.Text.Command(f"Compile [{self.cfg.master_path}]")
        finally:
            os.chdir(cwd)  # OpenDSS changes the working directory to the feeder folder
        self._lock_regulators(self.cfg.feeder.regulator_tap)
        # A meter at the substation defines the feeder zone, so OpenDSS can report each
        # node's distance from the substation. It has no electrical effect.
        dss.Text.Command("New EnergyMeter.substation element=Transformer.reg1a terminal=1")

        ones = " ".join(["1"] * HOURS)
        dss.Text.Command(f"New LoadShape.{LOAD_SHAPE} npts={HOURS} interval=1 mult=({ones})")
        dss.Text.Command(f"BatchEdit Load..* daily={LOAD_SHAPE}")

        pv = self.cfg.pv
        for name, bus, phase in self.sites:
            dss.Text.Command(f"New LoadShape.irr_{name} npts={HOURS} interval=1 mult=({ones})")
            dss.Text.Command(
                f"New PVSystem.{name} bus1={bus}.{phase} phases=1 kv={pv.kv} kVA={pv.kva} "
                f"Pmpp={pv.pmpp} pf=1 irradiance=1 daily=irr_{name} "
                f"%cutin={pv.cutin_pct} %cutout={pv.cutin_pct} kvarMax={pv.kva}"
            )
        dss.Text.Command("Set Mode=Daily stepsize=1h number=1")
        # A tight tolerance makes each solution independent of the previous one (the solver
        # warm-starts), so results don't depend on which policy or scenario ran before.
        dss.Text.Command("Set tolerance=1e-10")
        self.solve_at(12)

        self.node_names = [n.lower() for n in dss.Circuit.AllNodeNames()]
        index = {n: i for i, n in enumerate(self.node_names)}
        self.pv_node_idx = np.array([index[f"{bus}.{phase}"] for _, bus, phase in self.sites])
        self.node_distances = np.asarray(dss.Circuit.AllNodeDistances(), dtype=float)

    @staticmethod
    def _lock_regulators(tap: int) -> None:
        i = dss.RegControls.First()
        while i > 0:
            name = dss.RegControls.Name()
            dss.Text.Command(f"Edit RegControl.{name} TapNum={tap}")
            dss.Text.Command(f"Disable RegControl.{name}")
            i = dss.RegControls.Next()

    # -------------------------------------------------------------- scenario
    def set_profiles(self, profiles: DayProfiles) -> None:
        dss.LoadShape.Name(LOAD_SHAPE)
        dss.LoadShape.PMult(profiles.load.tolist())
        for i, name in enumerate(self.pv_names):
            dss.LoadShape.Name(f"irr_{name}")
            dss.LoadShape.PMult(profiles.irradiance[i].tolist())
        self._irradiance = profiles.irradiance.copy()

    def set_pv_enabled(self, enabled: bool) -> None:
        for name in self.pv_names:
            dss.Text.Command(f"{'Enable' if enabled else 'Disable'} PVSystem.{name}")

    def solve_at(self, hour: int) -> None:
        """Solve the power flow at ``hour`` (0..23).

        In Daily mode, ``Solve`` moves the clock forward one step after it runs. The hour
        is set again before every solve so repeated solves stay at the same time.
        """
        dss.Text.Command(f"Set hour={int(hour)}")
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError(f"OpenDSS power flow did not converge at hour {hour}")
        self.hour = int(hour)

    # -------------------------------------------------------------- inverters
    def available_power(self, hour: int) -> np.ndarray:
        """Maximum PV active power (kW) at ``hour`` given irradiance, limited by kVA."""
        return np.minimum(self._irradiance[:, hour] * self.pmpp, self.kva)

    def set_pv_setpoints(self, p_limit_kw: np.ndarray, q_kvar: np.ndarray) -> None:
        """Set each inverter's active-power ceiling (kW) and reactive-power output (kvar).

        Positive kvar injects reactive power and raises voltage. The ceiling is applied with
        ``%Pmpp``. OpenDSS applies irradiance itself, so available power is not counted twice.
        """
        for name, p, q in zip(self.pv_names, p_limit_kw, q_kvar):
            pct = float(np.clip(100.0 * p / self.pmpp, 0.0, 100.0))
            dss.Text.Command(f"Edit PVSystem.{name} %Pmpp={pct:.4f} kvar={float(q):.4f}")

    def reset_inverters(self) -> None:
        """Unity power factor, no curtailment."""
        for name in self.pv_names:
            dss.Text.Command(f"Edit PVSystem.{name} %Pmpp=100 pf=1")

    # ------------------------------------------------------------ measurement
    def node_voltages(self) -> np.ndarray:
        """Per-unit voltage magnitude of every node (bus phase), in ``self.node_names`` order."""
        return np.asarray(dss.Circuit.AllBusMagPu(), dtype=float)

    def pv_powers(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured terminal output of each PV (kW, kvar), positive = generation."""
        p = np.empty(self.n_pv)
        q = np.empty(self.n_pv)
        for i, name in enumerate(self.pv_names):
            dss.Circuit.SetActiveElement(f"PVSystem.{name}")
            s = dss.CktElement.Powers()
            p[i] = -sum(s[0::2])
            q[i] = -sum(s[1::2])
        return p, q
