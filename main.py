#!/usr/bin/env python3
"""BTC Race 24h bot: ACCUM-A + regime if-else (btc-race-24h-20261118-1444).

Entry: ACCUM-A strict entry, EXCEPT when the market is in an extreme
oversold regime — RSI(2) of the just-closed 15m bar < 10 — in which case
the LOOSE entry is used instead. The regime is read from current market
conditions at the decision bar (RSI(2) 15m), never from the window, bar
number, or price level: the same rule applies at any bar in any window.
  - STRICT: I2c: close < SMA90(1m) * (1 - 0.0075) AND RSI(2) 15m < 20.
    Stop = close * 0.99. #3c-flat (only if I2c did not fire): ATR(14,15m) /
    close < 0.0020 (20 bps) AND at least 2 of 4 legs. Stop = min(close *
    0.995, close - 1.5 * ATR).
  - LOOSE (only in the rsi2<10 regime): I2c dip 0.5% + RSI(2) 15m < 30,
    #3c-flat min 1 leg. Same stops.
  Why: on this competition's dev window (falling market, -2.80%) the strict
  entry never fires (0 trades, $0.00) while the loose entry fires once and
  loses -$0.52. The rsi2<10 regime delays the loose entry to a deeper dip
  (bar 135 instead of 105), turning it into a +$0.46 winner; threshold
  plateau 10-12 gives identical results.
Accumulation: buy 25% of capital ($250) per tranche; an extra 25% tranche
each time price drops 1% from the last tranche's fill price, max 4 tranches
(100% of capital). Exits (TP 3%, SL, trail, time 240 bars) are computed
from the qty-weighted average entry price; stop_px = avg_entry - stop_dist
(stop_dist fixed at the entry signal); accumulation is checked BEFORE the
SL; timestop A (240 bars counted from the FIRST buy). Long only, max 1
open position, no shorts, no leverage.

Indicator classes are copied VERBATIM from the backtest reference
(backtest_spike.py / backtest_spike2.py): same seeding, same smoothing,
same population variance in Bollinger, Wilder's ATR/RSI.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"
FEE_BPS = 4
SLIP_BPS = 2
TRANCHE_USD = 250.0
ACCUM_DROP = 0.01
MAX_TRANCHES = 4
TIME_STOP_BARS = 240
TP_PCT = 0.03
TRAIL_ACTIVATE = 0.02
TRAIL_GIVEBACK = 0.005
MAX_SLIP_BPS = 100


# ---------------------------------------------------------------------------
# Indicators (verbatim from backtest_spike.py / backtest_spike2.py)
# ---------------------------------------------------------------------------

class Ema:
    """House convention: value = mean of first span closes, then 2/(span+1) steps."""

    def __init__(self, span):
        self.span = span
        self.warm = []
        self.value = None

    def push(self, x):
        if self.value is not None:
            self.value += (2.0 / (self.span + 1)) * (x - self.value)
            return self.value
        self.warm.append(x)
        if len(self.warm) == self.span:
            self.value = sum(self.warm) / self.span
        return self.value


class Rsi:
    """Wilder's RSI(14): initial avg gain/loss over first 14 closes."""

    def __init__(self, span=14):
        self.span = span
        self.prev = None
        self.gains = []
        self.losses = []
        self.ag = None
        self.al = None
        self.value = None

    def push(self, x):
        if self.prev is None:
            self.prev = x
            return None
        ch = x - self.prev
        self.prev = x
        g = max(ch, 0.0)
        l = max(-ch, 0.0)
        if self.ag is None:
            self.gains.append(g)
            self.losses.append(l)
            if len(self.gains) == self.span:
                self.ag = sum(self.gains) / self.span
                self.al = sum(self.losses) / self.span
            return None
        self.ag = (self.ag * (self.span - 1) + g) / self.span
        self.al = (self.al * (self.span - 1) + l) / self.span
        self.value = 100.0 if self.al == 0 else 100.0 - 100.0 / (1.0 + self.ag / self.al)
        return self.value


class Bollinger:
    def __init__(self, span=20, mult=2.0):
        self.span = span
        self.mult = mult
        self.win = []

    def push(self, x):
        self.win.append(x)
        if len(self.win) > self.span:
            self.win.pop(0)
        if len(self.win) < self.span:
            return None
        m = sum(self.win) / self.span
        var = sum((v - m) ** 2 for v in self.win) / self.span
        sd = math.sqrt(var)
        return (m, m + self.mult * sd, m - self.mult * sd, 2 * self.mult * sd)


class Atr:
    def __init__(self, span=14):
        self.span = span
        self.prev_close = None
        self.trs = []
        self.value = None

    def push(self, h, l, c):
        if self.prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        self.prev_close = c
        if self.value is not None:
            self.value = (self.value * (self.span - 1) + tr) / self.span
            return self.value
        self.trs.append(tr)
        if len(self.trs) == self.span:
            self.value = sum(self.trs) / self.span
        return self.value


class Ind15:
    """All 15m indicators, updated once per completed 15m bar."""

    def __init__(self):
        self.ema12 = Ema(12)
        self.ema26 = Ema(26)
        self.ema9 = Ema(9)      # signal line on MACD
        self.ema20 = Ema(20)
        self.rsi = Rsi(14)
        self.bb = Bollinger(20, 2.0)
        self.atr = Atr(14)
        self.macd = None        # (line, signal, hist)
        self.macd_prev = None
        self.rsi_prev = None
        self.bb_hist = []       # bandwidth history for squeeze percentile
        self.closes = []
        self.bars15 = []        # (o, h, l, c) per completed 15m bar
        self.demand = []        # live demand zones
        self.supply = []        # live supply zones

    def _zones_update(self):
        n = len(self.bars15)
        # zones expire after 48 15m bars (12h)
        self.demand = [z for z in self.demand if n - z["bar"] <= 48]
        self.supply = [z for z in self.supply if n - z["bar"] <= 48]
        if n < 5:
            return
        # swing low at bar n-3 (lowest low of bars n-5..n-1, confirmed by 2 bars)
        j = n - 3
        lo = self.bars15[j][2]
        if lo == min(b[2] for b in self.bars15[j - 2:j + 3]):
            c_now = self.bars15[-1][3]
            if c_now >= lo * 1.006:  # strong rally away (>=0.6%)
                self.demand.append({"bottom": lo, "top": lo * 1.002,
                                    "bar": n})
        hi = self.bars15[j][1]
        if hi == max(b[1] for b in self.bars15[j - 2:j + 3]):
            c_now = self.bars15[-1][3]
            if c_now <= hi * 0.994:  # strong drop away (>=0.6%)
                self.supply.append({"bottom": hi * 0.998, "top": hi,
                                    "bar": n})

    def push(self, o, h, l, c):
        self.closes.append(c)
        self.bars15.append((o, h, l, c))
        self._zones_update()
        e12 = self.ema12.push(c)
        e26 = self.ema26.push(c)
        sig = None
        if e12 is not None and e26 is not None:
            line = e12 - e26
            sig = self.ema9.push(line)
            self.macd_prev = self.macd
            self.macd = (line, sig, line - sig if sig is not None else None)
        self.ema20.push(c)
        self.rsi_prev = self.rsi.value
        r = self.rsi.push(c)
        bb = self.bb.push(c)
        if bb is not None:
            self.bb_hist.append(bb[3])
            if len(self.bb_hist) > 20:
                self.bb_hist.pop(0)
        self.atr.push(h, l, c)
        return {"macd": self.macd, "macd_prev": self.macd_prev, "rsi": r,
                "rsi_prev": self.rsi_prev, "bb": bb, "atr": self.atr.value,
                "ema20": self.ema20.value}


class Ind15b(Ind15):
    """Ind15 + RSI(2), ATR-ratio, relative-bandwidth percentile, Donchian."""

    def __init__(self):
        super().__init__()
        self.rsi2 = Rsi(2)
        self.rsi2_prev = None
        self.atr_hist = []
        self.bw_rel_hist = []

    def push(self, o, h, l, c):
        d = super().push(o, h, l, c)
        self.rsi2_prev = self.rsi2.value
        d["rsi2"] = self.rsi2.push(c)
        d["rsi2_prev"] = self.rsi2_prev
        if self.atr.value is not None:
            self.atr_hist.append(self.atr.value)
        bb = d["bb"]
        if bb is not None:
            rel = bb[3] / bb[0] if bb[0] else 0.0
            self.bw_rel_hist.append(rel)
            d["bw_rel"] = rel
        else:
            d["bw_rel"] = None
        cl = self.closes
        d["sma5"] = sum(cl[-5:]) / 5 if len(cl) >= 5 else None
        d["sma20"] = sum(cl[-20:]) / 20 if len(cl) >= 20 else None
        d["atr_ratio"] = None
        if len(self.atr_hist) >= 20 and self.atr.value:
            sma_atr = sum(self.atr_hist[-20:]) / 20
            d["atr_ratio"] = self.atr.value / sma_atr if sma_atr else None
        # Donchian levels EXCLUDE the current bar (else unreachable).
        hs = [x[1] for x in self.bars15]
        ls = [x[2] for x in self.bars15]
        d["donch20_high"] = max(hs[-21:-1]) if len(hs) >= 21 else None
        d["donch10_low"] = min(ls[-11:-1]) if len(ls) >= 11 else None
        hist = self.bw_rel_hist
        if len(hist) >= 20:
            win = sorted(hist[-50:])
            d["bw_pct"] = sum(1 for v in win if v <= hist[-1]) / len(win)
        else:
            d["bw_pct"] = None
        return d


# ---------------------------------------------------------------------------
# Entry functions (verbatim from backtest_spike2.py / backtest_accum_loose.py)
# ---------------------------------------------------------------------------

def make_e3(min_legs=1, stop_pct=0.005, stop_atr=1.5, rsi_mode="rsi2",
            pure_atr=False):
    """OR-union with tunable confluence (min legs) and stop width.
    min_legs=1, stop 0.5%/1.5xATR reproduces the original e3_or."""
    def e3(ind, m15, b, ctx):
        bb = m15["bb"]
        r2 = m15["rsi2"]
        n = 0
        if bb is not None and b["c"] < bb[2]:
            n += 1
        if rsi_mode == "rsi14":
            r14 = m15["rsi"]  # Ind15.rsi is RSI(14)
            if r14 is not None and r14 < 30:
                n += 1
        elif r2 is not None and r2 < 10:
            n += 1
        if m15["ema20"] is not None and ind.atr.value:
            z = (b["c"] - m15["ema20"]) / ind.atr.value
            if z < -2.0:
                n += 1
        if m15["ema20"] is not None and ctx["trend"]:
            cl = ind.closes
            if len(cl) >= 2 and cl[-2] <= m15["ema20"] and cl[-1] > m15["ema20"]:
                n += 1
        if n >= min_legs:
            atr = ind.atr.value
            if pure_atr and atr:
                return b["c"] - 2.0 * atr
            stop = b["c"] * (1 - stop_pct)
            if atr:
                stop = min(stop, b["c"] - stop_atr * atr)
            return stop
        return None
    return e3


def make_e4(flush_bps=75, gate="none", sl_pct=0.01, trend_gate=False):
    """Candidate #4 (LB pattern): entry = 1m close flush_bps below SMA90(1m).
    gate: 'none' | 'bounce' (just-closed 15m bar closed above its open) |
          'rsi2' (RSI(2) < 20, deep oversold).
    trend_gate: require ctx["trend"] (EMA60 rising + price above EMA60) —
    the research long-trend gate, tested to cut drawdown ~1/3.
    Stop = entry * (1 - sl_pct)."""
    def e4(ind, m15, b, ctx):
        sma90 = ctx.get("sma90_1m")
        if sma90 is None:
            return None
        if b["c"] >= sma90 * (1 - flush_bps / 10000.0):
            return None
        if trend_gate and not ctx.get("trend"):
            return None
        if gate == "bounce":
            bars = ind.bars15
            if len(bars) < 1 or bars[-1][3] <= bars[-1][0]:
                return None
        elif gate == "rsi2":
            r2 = m15["rsi2"]
            if r2 is None or r2 >= 20:
                return None
        return b["c"] * (1 - sl_pct)
    return e4


def make_combo_flat(atr_thr_bps=20.0):
    """Identical to run_combo_flat.make_combo_flat: I2c first, then #3c-flat
    only when ATR(14,15m) < 20 bps. Returns the stop price (float) or None."""
    e_i2c = make_e4(75, "rsi2", 0.01)
    e_3c = make_e3(2, 0.005, 1.5, "rsi2", False)

    def combo(ind, m15, b, ctx):
        s = e_i2c(ind, m15, b, ctx)
        if s is not None:
            return s
        atr = ind.atr.value
        if atr is not None and atr / b["c"] * 10000 < atr_thr_bps:
            return e_3c(ind, m15, b, ctx)
        return None

    return combo


def make_e4_loose(flush_bps=50, rsi_thr=30, sl_pct=0.01):
    """Verbatim copy of backtest_accum_loose.make_e4_loose: same as make_e4
    except the RSI(2) gate threshold is a parameter (30, not hardcoded 20)."""
    def e4(ind, m15, b, ctx):
        sma90 = ctx.get("sma90_1m")
        if sma90 is None:
            return None
        if b["c"] >= sma90 * (1 - flush_bps / 10000.0):
            return None
        r2 = m15["rsi2"]
        if r2 is None or r2 >= rsi_thr:
            return None
        return b["c"] * (1 - sl_pct)
    return e4


def make_combo_flat_loose(atr_thr_bps=20.0):
    """Loose entry: I2c dip 0.5% + RSI(2) < 30, #3c-flat min 1 leg."""
    e_i2c = make_e4_loose(50, 30, 0.01)
    e_3c = make_e3(1, 0.005, 1.5, "rsi2", False)

    def combo(ind, m15, b, ctx):
        s = e_i2c(ind, m15, b, ctx)
        if s is not None:
            return s
        atr = ind.atr.value
        if atr is not None and atr / b["c"] * 10000 < atr_thr_bps:
            return e_3c(ind, m15, b, ctx)
        return None

    return combo


def make_accum_regime(rsi2_thr=10.0, atr_thr_bps=20.0):
    """Regime if-else entry: when RSI(2) of the just-closed 15m bar is
    below rsi2_thr (extreme oversold regime) use the LOOSE entry, otherwise
    use the STRICT ACCUM-A entry. The regime is read from current market
    conditions only — never from the window, bar number, or price level."""
    strict = make_combo_flat(atr_thr_bps)
    loose = make_combo_flat_loose(atr_thr_bps)

    def entry(ind, m15, b, ctx):
        r2 = m15.get("rsi2")
        if r2 is not None and r2 < rsi2_thr:
            return loose(ind, m15, b, ctx)
        return strict(ind, m15, b, ctx)

    return entry


# ---------------------------------------------------------------------------
# Bot state
# ---------------------------------------------------------------------------

class Bot:
    def __init__(self, hello):
        sym = hello["symbols"][0]
        self.price_scale = int(sym["price_scale"])
        self.qty_scale = int(sym["qty_scale"])
        self.quote_scale = int(hello["capital"]["quote_scale"])
        self.tranche_raw = int(TRANCHE_USD * (10 ** self.quote_scale))
        self.ema60 = Ema(60)
        self.ema60_hist = []
        self.ind = Ind15b()
        self.closes15 = []
        self.h15 = []
        self.l15 = []
        self.c1m = []
        self.trend = False
        self.sma90 = None
        # position tracking
        self.tranches = []       # {qty_btc, fill_px, fill_seq}
        self.stop_dist = 0.0
        self.stop_px = 0.0
        self.highest = 0.0
        self.entry_seq_first = -1
        # pending order placed on the previous batch (to match its fill)
        self.pending = None      # {"side","client_id","notional_raw","dec_close"}
        self.pending_stop_dist = 0.0
        self.order_no = 0
        self.entry_fn = make_accum_regime(10.0, 20.0)

    def _usd(self, raw):
        return int(raw) * (10 ** (-self.price_scale))

    def _avg_entry(self):
        tq = sum(t["qty_btc"] for t in self.tranches)
        if tq <= 0:
            return 0.0
        return sum(t["qty_btc"] * t["fill_px"] for t in self.tranches) / tq

    def _next_id(self, prefix):
        self.order_no += 1
        return "%s%d" % (prefix, self.order_no)

    def _process_fills(self, state, batch, bar):
        """Fold the fills of the previous batch's order into the tranches.
        Uses the fill's own price/qty (exact, from the harness); falls back
        to the SPEC inference formula when a price is missing."""
        fills = batch.get("fills") or []
        if self.pending is None:
            return
        pend = self.pending
        self.pending = None
        for f in fills:
            if f.get("client_id") != pend["client_id"]:
                continue
            if f.get("status") != "filled":
                sys.stderr.write("fill rejected %s reason=%s seq=%d\n"
                                 % (pend["side"], f.get("reason"), state.seq))
                return
            if pend["side"] == "buy":
                price_raw = f.get("price")
                qty_raw = f.get("qty")
                if price_raw is not None and int(price_raw) > 0:
                    fill_px = self._usd(price_raw)
                else:
                    # SPEC inference: open * (1+slip) * (1+a/depth)
                    a = (pend["notional_raw"] / (10 ** self.quote_scale)) * (1 - FEE_BPS / 10000.0)
                    depth = bar["o_depth_usd"]
                    fill_px = bar["o_usd"] * (1 + SLIP_BPS / 10000.0) * (1 + a / depth) if depth else bar["o_usd"]
                if qty_raw is not None and int(qty_raw) > 0:
                    qty_btc = int(qty_raw) * (10 ** (-self.qty_scale))
                else:
                    a = (pend["notional_raw"] / (10 ** self.quote_scale)) * (1 - FEE_BPS / 10000.0)
                    qty_btc = a / fill_px if fill_px else 0.0
                self.tranches.append({"qty_btc": qty_btc, "fill_px": fill_px,
                                      "fill_seq": state.seq})
                if len(self.tranches) == 1:
                    self.entry_seq_first = state.seq
                    self.stop_dist = self.pending_stop_dist
                    self.highest = bar["h_usd"]
                # recompute the stop from the new average entry
                self.stop_px = self._avg_entry() - self.stop_dist
                sys.stderr.write("buy filled seq=%d px=%.2f qty=%.6f tranches=%d avg=%.2f stop=%.2f\n"
                                 % (state.seq, fill_px, qty_btc, len(self.tranches),
                                    self._avg_entry(), self.stop_px))
            else:  # sell filled: position closed
                sys.stderr.write("sell filled seq=%d\n" % state.seq)
                self.tranches = []
                self.stop_dist = 0.0
                self.stop_px = 0.0
                self.highest = 0.0
                self.entry_seq_first = -1
            return

    def on_batch(self, state, batch):
        bars = batch.get("bars") or []
        bar = None
        for b in bars:
            if b.get("symbol") == SYMBOL:
                bar = b
                break
        if bar is None:
            return []
        o_usd = self._usd(bar["o"])
        h_usd = self._usd(bar["h"])
        l_usd = self._usd(bar["l"])
        c_usd = self._usd(bar["c"])
        o_depth_usd = int(bar["o_depth"][0]) / 1e6 if bar.get("o_depth") else 0.0
        bar_u = {"o_usd": o_usd, "h_usd": h_usd, "l_usd": l_usd, "c_usd": c_usd,
                 "o_depth_usd": o_depth_usd}

        # 1) fills of the previous batch's order (at this bar's open)
        self._process_fills(state, batch, bar_u)

        # 2) update indicators with this bar's close
        e60 = self.ema60.push(c_usd)
        self.ema60_hist.append(e60)
        self.closes15.append(c_usd)
        self.h15.append(h_usd)
        self.l15.append(l_usd)
        self.c1m.append(c_usd)
        m15 = None
        if len(self.closes15) == 15:
            m15 = self.ind.push(self.closes15[0], max(self.h15),
                                min(self.l15), self.closes15[-1])
            self.closes15, self.h15, self.l15 = [], [], []
        trend = False
        if e60 is not None and len(self.ema60_hist) > 60:
            prev60 = self.ema60_hist[-61]
            trend = c_usd > e60 and prev60 is not None and e60 > prev60
        self.trend = trend
        self.sma90 = (sum(self.c1m[-90:]) / 90 if len(self.c1m) >= 90 else None)
        ctx = {"trend": self.trend, "sma90_1m": self.sma90}

        # 3) decisions at this bar's close
        holding = len(self.tranches) > 0
        if holding:
            self.highest = max(self.highest, h_usd)
            avg = self._avg_entry()
            # 3a) accumulation FIRST (see backtest docstring for why)
            if (len(self.tranches) < MAX_TRANCHES
                    and state.cash >= self.tranche_raw
                    and c_usd <= self.tranches[-1]["fill_px"] * (1 - ACCUM_DROP)):
                cid = self._next_id("b")
                self.pending = {"side": "buy", "client_id": cid,
                                "notional_raw": self.tranche_raw,
                                "dec_close": c_usd}
                return [{"side": "buy", "symbol": SYMBOL, "client_id": cid,
                         "notional_quote_units": self.tranche_raw,
                         "max_slippage_bps": MAX_SLIP_BPS}]
            # 3b) exits, in order: TP, SL, trail, time
            reason = None
            if c_usd >= avg * (1 + TP_PCT):
                reason = "tp"
            elif c_usd <= self.stop_px:
                reason = "sl"
            elif (self.highest >= avg * (1 + TRAIL_ACTIVATE)
                    and c_usd <= self.highest - TRAIL_GIVEBACK * avg):
                reason = "trail"
            elif state.seq - self.entry_seq_first >= TIME_STOP_BARS:
                reason = "time"
            if reason is not None:
                held = state.positions.get(SYMBOL)
                if held is None or held.qty <= 0:
                    return []
                cid = self._next_id("s") + "-" + reason
                self.pending = {"side": "sell", "client_id": cid,
                                "notional_raw": 0, "dec_close": c_usd}
                sys.stderr.write("exit %s seq=%d\n" % (reason, state.seq))
                return [{"side": "sell", "symbol": SYMBOL, "client_id": cid,
                         "qty": held.qty, "max_slippage_bps": MAX_SLIP_BPS}]
            return []
        # flat: entry only when not warming up and a 15m bar just completed
        if state.warming_up or m15 is None:
            return []
        b = {"c": c_usd}
        sig = self.entry_fn(self.ind, m15, b, ctx)
        if sig is not None and isinstance(sig, float):
            if state.cash < self.tranche_raw:
                return []
            cid = self._next_id("b")
            self.pending = {"side": "buy", "client_id": cid,
                            "notional_raw": self.tranche_raw,
                            "dec_close": c_usd}
            self.pending_stop_dist = c_usd - sig
            return [{"side": "buy", "symbol": SYMBOL, "client_id": cid,
                     "notional_quote_units": self.tranche_raw,
                     "max_slippage_bps": MAX_SLIP_BPS}]
        return []


_BOT = [None]


def on_batch(state, batch):
    if _BOT[0] is None:
        _BOT[0] = Bot(state.hello)
    return _BOT[0].on_batch(state, batch)


def main():
    makefun_bars.run(on_batch)


if __name__ == "__main__":
    main()
