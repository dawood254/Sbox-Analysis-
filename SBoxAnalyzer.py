"""
S-Box Cryptographic Analyzer
Single-file PyQt5 application.

Supports arbitrary n→m S-boxes (3-bit to 8-bit input, any output width).
Metrics: Nonlinearity · SAC · BIC-SAC · LAT · DDT ·
         Balancedness · Linear Structure · Autocorrelation · Hybrid LAT/DDT
"""

# ===================================================================
#  SECTION 1 — CRYPTOGRAPHIC CORE
# ===================================================================
import sys, os, math, datetime
from collections import Counter
import numpy as np


def parse_sbox(text: str) -> list:
    """
    Parse an S-box from text.  Accepts commas, spaces, tabs, newlines,
    semicolons, brackets, and hex (0x…) notation.
    """
    text = text.strip()
    for ch in "{}[]()":
        text = text.replace(ch, " ")
    for ch in "\n\r;\t,":
        text = text.replace(ch, " ")
    values = []
    for tok in text.split():
        tok = tok.strip()
        if not tok:
            continue
        try:
            if tok.lower().startswith("0x"):
                values.append(int(tok, 16))
            else:
                values.append(int(tok))
        except ValueError:
            raise ValueError(
                f"Invalid token in S-box input: {tok!r}.\n"
                "Invalid S-box. Use decimal or hexadecimal integers separated by "
                "commas, spaces, tabs, newlines, or semicolons."
            )
    N = len(values)
    if N == 0:
        raise ValueError(
            "No numeric values found.\n"
            "Invalid S-box. The number of entries must be a power of two between 8 and 256.\n"
            "Example: 64 entries for DES 6→4, 256 entries for AES 8→8."
        )
    log2N = math.log2(N)
    if abs(log2N - round(log2N)) > 1e-9 or N < 8 or N > 256:
        raise ValueError(
            f"Got {N} entries — not a power of 2 between 8 and 256.\n"
            "Invalid S-box. The number of entries must be a power of two between 8 and 256.\n"
            "Example: 64 entries for DES 6→4, 256 entries for AES 8→8."
        )
    if any(v < 0 for v in values):
        raise ValueError("All S-box values must be ≥ 0.")
    return values


def _wht(arr: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform (in-place algorithm on a copy)."""
    result = np.array(arr, dtype=np.float64)
    N = len(result)
    step = 1
    while step < N:
        for i in range(0, N, step * 2):
            lo = result[i:i + step].copy()
            hi = result[i + step:i + 2 * step].copy()
            result[i:i + step]             = lo + hi
            result[i + step:i + 2 * step] = lo - hi
        step *= 2
    return result


class SBoxCore:
    def __init__(self, sbox: list):
        self.sbox     = np.array(sbox, dtype=np.int32)
        self.N        = len(sbox)
        self.n        = int(round(math.log2(self.N)))   # input bits  (3–8)
        max_val       = int(max(sbox))
        self.M        = max_val + 1
        self.m        = max(1, math.ceil(math.log2(self.M)))  # output bits
        self.out_size = 1 << self.m                      # 2^m
        self._validate()

    def _validate(self):
        log2N = math.log2(self.N)
        if abs(log2N - round(log2N)) > 1e-9 or self.N < 8 or self.N > 256:
            raise ValueError(
                f"S-box must have 2^n entries for n in [3,8]. Got {self.N}.\n"
                "Invalid S-box. The number of entries must be a power of two between 8 and 256.\n"
                "Example: 64 entries for DES 6→4, 256 entries for AES 8→8."
            )
        if self.n < 3 or self.n > 8:
            raise ValueError(
                f"Input bits n={self.n} out of range [3,8]. Got {self.N} entries."
            )
        if self.sbox.min() < 0:
            raise ValueError(f"All values must be ≥ 0. Found {self.sbox.min()}.")
        if self.sbox.max() >= self.out_size:
            raise ValueError(
                f"Values must be < {self.out_size}. Found {self.sbox.max()}."
            )

    def _comp(self, b: int) -> np.ndarray:
        """Component function f_b(x) = popcount(b & S[x]) mod 2."""
        return np.array(
            [bin(int(b) & int(s)).count("1") % 2 for s in self.sbox],
            dtype=np.int32
        )

    def _walsh(self, f01: np.ndarray) -> np.ndarray:
        """Walsh transform of a ±1-valued sequence derived from f01 ∈ {0,1}^N."""
        return _wht((-1.0) ** f01)

    # ------------------------------------------------------------------
    # 1. NONLINEARITY
    #    b ranges over 1 … 2^m-1  (all nonzero output masks)
    # ------------------------------------------------------------------
    def nonlinearity(self):
        N, n, m = self.N, self.n, self.m
        out_size = self.out_size
        nl_min = N
        nl_max = 0
        nl_sum = 0.0
        lat_max_M = 0
        num_components = out_size - 1    # b = 1 … 2^m - 1
        comp_table = []

        for b in range(1, out_size):
            W = self._walsh(self._comp(b))
            walsh_max = int(np.max(np.abs(W)))
            nl_b = N // 2 - walsh_max // 2
            if walsh_max > lat_max_M:
                lat_max_M = walsh_max
            if nl_b < nl_min:
                nl_min = nl_b
            if nl_b > nl_max:
                nl_max = nl_b
            nl_sum += nl_b
            bin_str = format(b, f"0{m}b")
            comp_table.append({
                "b": b,
                "binary": bin_str,
                "walsh_max": walsh_max,
                "nl": nl_b,
            })

        nl_vectorial = N // 2 - lat_max_M // 2
        nl_avg = round(nl_sum / num_components, 4)

        return {
            "nl_vectorial":     nl_vectorial,
            "nl_min":           nl_min,
            "nl_max":           nl_max,
            "nl_avg":           nl_avg,
            "lat_max_M":        lat_max_M,
            "num_components":   num_components,
            "comp_table":       comp_table,
        }

    # ------------------------------------------------------------------
    # 2. SAC  (n × m matrix)
    # ------------------------------------------------------------------
    def sac(self):
        N, n, m = self.N, self.n, self.m
        idx = np.arange(N, dtype=np.int32)
        mat = np.zeros((n, m))
        for i in range(n):
            diffs = self.sbox ^ self.sbox[idx ^ (1 << i)]
            for j in range(m):
                mat[i, j] = float(np.sum((diffs >> j) & 1)) / N

        return {
            "matrix":           mat.tolist(),
            "n_rows":           n,
            "n_cols":           m,
            "avg":              round(float(np.mean(mat)), 6),
            "min":              round(float(np.min(mat)), 6),
            "max":              round(float(np.max(mat)), 6),
            "passes_criterion": bool(np.all(np.abs(mat - 0.5) < 0.1)),
            "deviation_max":    round(float(np.max(np.abs(mat - 0.5))), 6),
        }

    # ------------------------------------------------------------------
    # 3. BIC-SAC  (m × m matrix)
    # ------------------------------------------------------------------
    def bic_sac(self):
        N, n, m = self.N, self.n, self.m
        idx = np.arange(N, dtype=np.int32)
        mat = np.full((m, m), np.nan)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                acc = 0.0
                for k in range(n):
                    diffs = self.sbox ^ self.sbox[idx ^ (1 << k)]
                    acc  += float(np.sum(((diffs >> i) & 1) ^ ((diffs >> j) & 1))) / N
                mat[i, j] = acc / n

        valid = mat[~np.isnan(mat)]
        return {
            "matrix":           mat.tolist(),
            "n_size":           m,
            "avg":              round(float(np.nanmean(mat)), 6),
            "min":              round(float(np.nanmin(mat)), 6),
            "max":              round(float(np.nanmax(mat)), 6),
            "passes_criterion": bool(np.all(np.abs(valid - 0.5) < 0.1)),
            "deviation_max":    round(float(np.max(np.abs(valid - 0.5))), 6),
        }

    # ------------------------------------------------------------------
    # 4. LAT  (N × out_size table)
    #    rows  = input masks  a = 0 … 2^n - 1
    #    cols  = output masks b = 0 … 2^m - 1
    # ------------------------------------------------------------------
    def lat(self):
        N = self.N
        out_size = self.out_size
        table = np.zeros((N, out_size), dtype=np.int32)
        table[:, 0] = np.round(self._walsh(np.zeros(N, dtype=np.int32))).astype(np.int32)
        for b in range(1, out_size):
            table[:, b] = np.round(self._walsh(self._comp(b))).astype(np.int32)

        sub = table[1:, 1:]        # exclude trivial input/output masks
        max_abs = int(np.max(np.abs(sub)))
        lp_max  = (max_abs / (N // 2)) ** 2 if N > 1 else 0.0
        nl_from_lat = N // 2 - max_abs // 2
        nz_vals = np.abs(sub)[np.abs(sub) > 0]
        min_nonzero_abs = int(np.min(nz_vals)) if nz_vals.size else 0

        flat = np.abs(sub).ravel()
        n_top = min(10, flat.size)
        top_idx = flat.argsort()[-n_top:][::-1]
        top10 = []
        for fi in top_idx:
            a_idx  = int(fi // (out_size - 1)) + 1
            b_idx  = int(fi % (out_size - 1)) + 1
            top10.append({
                "a":     a_idx,
                "b":     b_idx,
                "value": int(sub.ravel()[fi]),
            })

        return {
            "table":        table.tolist(),
            "max_abs":      max_abs,
            "min_nonzero_abs": min_nonzero_abs,
            "lp_max":       round(lp_max, 8),
            "nl_from_lat":  nl_from_lat,
            "shape":        f"{N}×{out_size}",
            "top10":        top10,
        }

    # ------------------------------------------------------------------
    # 5. DDT  (N × out_size table)
    #    rows  = input differences  Δx = 0 … 2^n - 1
    #    cols  = output differences Δy = 0 … 2^m - 1
    # ------------------------------------------------------------------
    def ddt(self):
        N = self.N
        out_size = self.out_size
        idx   = np.arange(N, dtype=np.int32)
        table = np.zeros((N, out_size), dtype=np.int32)
        table[0, 0] = N
        for dx in range(1, N):
            output_diffs = (self.sbox ^ self.sbox[idx ^ dx]).astype(np.int32)
            for od in output_diffs:
                table[dx, int(od)] += 1

        sub = table[1:, :]         # exclude dx=0 row
        du  = int(np.max(sub))
        nz_vals = sub[sub > 0]
        min_nonzero = int(np.min(nz_vals)) if nz_vals.size else 0

        flat  = sub.ravel()
        n_top = min(10, flat.size)
        top_idx = flat.argsort()[-n_top:][::-1]
        top10 = []
        for fi in top_idx:
            dx_idx = int(fi // out_size) + 1
            dy_idx = int(fi % out_size)
            top10.append({
                "a":     dx_idx,
                "b":     dy_idx,
                "count": int(flat[fi]),
            })

        return {
            "table":  table.tolist(),
            "du":     du,
            "dp":     round(du / N, 8),
            "min_nonzero": min_nonzero,
            "shape":  f"{N}×{out_size}",
            "top10":  top10,
        }

    # ------------------------------------------------------------------
    # 6. BALANCEDNESS
    # ------------------------------------------------------------------
    def balancedness(self):
        N, n, m = self.N, self.n, self.m
        out_size = self.out_size
        counts = Counter(int(v) for v in self.sbox)

        # Bijective only if n==m AND it's a permutation
        is_bijective = (n == m) and (len(counts) == N) and all(v == 1 for v in counts.values())

        coord_info = []
        for j in range(m):
            ones  = int(np.sum((self.sbox >> j) & 1))
            zeros = N - ones
            coord_info.append({
                "bit":      j,
                "ones":     ones,
                "zeros":    zeros,
                "balanced": ones == N // 2,
            })

        return {
            "is_bijective":       is_bijective,
            "distinct_outputs":   len(counts),
            "input_size":         N,
            "output_size":        out_size,
            "all_coords_balanced": all(c["balanced"] for c in coord_info),
            "coord_info":         coord_info,
            "max_freq":           max(counts.values()),
            "min_freq":           min(counts.values()),
        }

    # ------------------------------------------------------------------
    # 7. LINEAR STRUCTURE
    # ------------------------------------------------------------------
    def linear_structure(self):
        N, n, m = self.N, self.n, self.m
        idx = np.arange(N, dtype=np.int32)
        by_bit = {}
        bits_with_ls = 0

        for j in range(m):
            f_j = (self.sbox >> j) & 1
            ls_list = []
            for u in range(1, N):
                d = f_j ^ f_j[idx ^ u]
                if np.all(d == d[0]):
                    ls_list.append({"u": int(u), "constant": int(d[0])})
            has_ls = len(ls_list) > 0
            if has_ls:
                bits_with_ls += 1
            by_bit[f"out_bit_{j}"] = {
                "count":      len(ls_list),
                "structures": ls_list[:10],
                "has_ls":     has_ls,
            }

        return {
            "by_bit":       by_bit,
            "bits_with_ls": bits_with_ls,
            "total_bits":   m,
            "has_any_ls":   bits_with_ls > 0,
        }

    # ------------------------------------------------------------------
    # 8. AUTOCORRELATION
    # ------------------------------------------------------------------
    def autocorrelation(self):
        N, n, m = self.N, self.n, self.m
        by_bit  = {}
        gmx     = 0.0

        for j in range(m):
            f  = ((self.sbox >> j) & 1).astype(np.float64)
            W  = _wht((-1.0) ** f)
            ac = _wht(W ** 2) / N
            mx = float(np.max(np.abs(ac[1:])))
            if mx > gmx:
                gmx = mx
            by_bit[f"out_bit_{j}"] = {
                "max_abs": round(mx, 4),
                "ac0":     round(float(ac[0]), 2),
            }

        return {
            "by_bit":     by_bit,
            "abac":       round(gmx, 4),
            "global_max": round(gmx, 4),
        }

    # ------------------------------------------------------------------
    # 9. HYBRID LAT/DDT
    # ------------------------------------------------------------------
    def hybrid_lat_ddt(self):
        N, n, m = self.N, self.n, self.m
        out_size = self.out_size

        gmx = 0
        Ws  = []
        for b in range(1, out_size):
            W = self._walsh(self._comp(b))
            mx = int(np.max(np.abs(W)))
            if mx > gmx:
                gmx = mx
            Ws.append(W)

        nl_vectorial = N // 2 - gmx // 2
        lp_max = (gmx / (N // 2)) ** 2 if N > 1 else 0.0

        idx = np.arange(N, dtype=np.int32)
        du  = 0
        for dx in range(1, N):
            c = Counter(int(v) for v in (self.sbox ^ self.sbox[idx ^ dx]))
            local_max = c.most_common(1)[0][1]
            if local_max > du:
                du = local_max
        dp = du / N

        is_bijective = (n == m) and (len(set(int(v) for v in self.sbox)) == N)
        if is_bijective:
            sign_vec = np.where(np.arange(N) & 1, -1.0, 1.0)
            total    = sum(np.dot(W ** 2, sign_vec) for W in Ws)
            exp_val  = -(N ** 2)
            bridge_check = (
                f"PASS  (expected {exp_val}, got {int(round(total))})"
                if abs(total - exp_val) < 1.0
                else f"FAIL  (expected {exp_val}, got {int(round(total))})"
            )
        else:
            bridge_check = "N/A"

        return {
            "nl_vectorial": nl_vectorial,
            "lat_max":      gmx,
            "lp_max":       round(lp_max, 8),
            "du":           du,
            "dp":           round(dp, 8),
            "bridge_check": bridge_check,
            "nl_formula":   f"NL_vec = {N//2} - {gmx}//2 = {nl_vectorial}",
            "dp_formula":   f"DP_max = {du}/{N} = {dp:.8f}",
        }

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------
    def run_all(self, cb=None):
        steps = [
            ("nonlinearity",     "Nonlinearity"),
            ("sac",              "SAC"),
            ("bic_sac",          "BIC-SAC"),
            ("lat",              "LAT (full table)"),
            ("ddt",              "DDT (full table)"),
            ("balancedness",     "Balancedness"),
            ("linear_structure", "Linear Structure"),
            ("autocorrelation",  "Autocorrelation"),
            ("hybrid_lat_ddt",   "Hybrid LAT/DDT"),
        ]
        results = {}
        for i, (method, label) in enumerate(steps):
            if cb:
                cb(i, len(steps), label)
            results[method] = getattr(self, method)()
        if cb:
            cb(len(steps), len(steps), "Done")
        return results


# ===================================================================
#  SECTION 2 — TEXT FORMATTERS
# ===================================================================

def _H(t):
    return f"\n{'='*62}\n  {t}\n{'='*62}"

def _S(t):
    return f"\n{'─'*50}\n  {t}\n{'─'*50}"

def _B(v):
    return "Yes" if v else "No"

def _mat_txt(mat, row_labels, col_labels, fmt=".4f"):
    W = 9
    header = " " * 10 + "".join(f"{c:>{W}}" for c in col_labels)
    sep    = " " * 10 + "─" * (W * len(col_labels))
    lines  = [header, sep]
    for r_lbl, row in zip(row_labels, mat):
        cells = []
        for v in row:
            if isinstance(v, float) and math.isnan(v):
                cells.append(f"{'---':>{W}}")
            else:
                cells.append(f"{v:>{W}{fmt}}")
        lines.append(f"  {r_lbl:<8}" + "".join(cells))
    return "\n".join(lines)


def fmt_summary(r, sbox, n_in, m_out):
    nl  = r["nonlinearity"]
    sac = r["sac"]
    bic = r["bic_sac"]
    bal = r["balancedness"]
    ls  = r["linear_structure"]
    ac  = r["autocorrelation"]
    hyb = r["hybrid_lat_ddt"]
    lat = r["lat"]
    ddt = r["ddt"]
    N   = len(sbox)

    sac_crit = "PASS" if sac["passes_criterion"] else "FAIL"
    bic_crit = "PASS" if bic["passes_criterion"] else "FAIL"

    W = 20

    def row(label, value):
        return f"  {label:<28}{str(value):<{W}}"

    sep = "  " + "─" * 46

    lines = [
        _H("S-BOX ANALYSIS SUMMARY"),
        "",
        f"  S-BOX TYPE:   {n_in}→{m_out}  ({n_in}-bit input, {m_out}-bit output)",
        f"  ENTRIES:      {N}",
        f"  MAX OUTPUT:   {int(max(sbox))}",
        f"  BIJECTIVE:    {_B(bal['is_bijective'])}",
        "",
        f"  {'METRIC':<28}{'VALUE':<{W}}",
        sep,
        row("Min NL",           nl["nl_min"]),
        row("Max NL",           nl["nl_max"]),
        row("Avg NL",           nl["nl_avg"]),
        row("NL vectorial",     nl["nl_vectorial"]),
        row("LAT max M(S)",     nl["lat_max_M"]),
        row("LP_max",           f"{hyb['lp_max']:.8f}"),
        sep,
        row("DU (diff. unif.)", hyb["du"]),
        row("DP_max",           f"{hyb['dp']:.8f}"),
        sep,
        row("SAC average",      f"{sac['avg']:.6f}"),
        row("SAC criterion",    sac_crit),
        row("BIC-SAC average",  f"{bic['avg']:.6f}"),
        row("BIC-SAC criterion",bic_crit),
        sep,
        row("Bijective",        _B(bal["is_bijective"])),
        row("All bits balanced",_B(bal["all_coords_balanced"])),
        row("ABAC",             ac["abac"]),
        row("Bridge check",     hyb["bridge_check"]),
        sep,
    ]
    return "\n".join(lines)


def fmt_comp_nl(r):
    nl  = r["nonlinearity"]
    tbl = nl["comp_table"]
    m_bits = len(tbl[0]["binary"]) if tbl else 8

    lines = [
        _H("COMPONENT NONLINEARITY  (all output masks b = 1 … 2^m-1)"),
        f"\n  Vectorial NL  NL_vec   :  {nl['nl_vectorial']}",
        f"  NL  min / avg / max    :  {nl['nl_min']} / {nl['nl_avg']} / {nl['nl_max']}",
        f"  LAT max  M(S)          :  {nl['lat_max_M']}",
        f"  Number of components   :  {nl['num_components']}  (= 2^m - 1)",
        _S("Component table"),
        f"  {'Mask b':>8}  {'Binary':>{m_bits+2}}  {'Walsh Max':>10}  {'NL':>6}",
        "  " + "─" * (8 + m_bits + 2 + 10 + 6 + 12),
    ]
    for e in tbl:
        lines.append(
            f"  {e['b']:>8}  {e['binary']:>{m_bits+2}}  {e['walsh_max']:>10}  {e['nl']:>6}"
        )
    lines += [
        _S("Interpretation"),
        "  NL_vec is the minimum NL over ALL 2^m-1 non-zero output masks.",
        "  Higher NL_vec = stronger linear cryptanalysis resistance.",
        "  The vectorial NL is typically ≤ all individual mask NLs.",
    ]
    return "\n".join(lines)


def fmt_sac(r):
    sac = r["sac"]
    n   = sac["n_rows"]
    m   = sac["n_cols"]
    return "\n".join([
        _H(f"SAC  —  STRICT AVALANCHE CRITERION  [{n}×{m} matrix: input bits × output bits]"),
        f"\n  Criterion passes  :  {_B(sac['passes_criterion'])}",
        f"  Average           :  {sac['avg']:.6f}   (ideal = 0.500000)",
        f"  Min  /  Max       :  {sac['min']:.6f}  /  {sac['max']:.6f}",
        f"  Max |entry−0.5|   :  {sac['deviation_max']:.6f}",
        f"  Matrix shape      :  {n} rows (input bits)  ×  {m} cols (output bits)",
        _S("SAC Matrix  [row = flipped input bit i,  col = output bit j]"),
        "  SAC[i][j] = Pr_x[ bit_j(S(x) XOR S(x XOR 2^i)) = 1 ]   ideal = 0.5\n",
        _mat_txt(
            sac["matrix"],
            [f"In{i}" for i in range(n)],
            [f"Out{j}" for j in range(m)],
        ),
        _S("Interpretation"),
        "  Green ≈ 0.5 (good).  Orange = borderline.  Red = weakness.",
    ])


def fmt_bic_sac(r):
    bic = r["bic_sac"]
    m   = bic["n_size"]
    return "\n".join([
        _H(f"BIC-SAC  —  BIT INDEPENDENCE CRITERION  [{m}×{m} matrix: output bits × output bits]"),
        f"\n  Criterion passes  :  {_B(bic['passes_criterion'])}",
        f"  Average           :  {bic['avg']:.6f}   (ideal = 0.500000)",
        f"  Min  /  Max       :  {bic['min']:.6f}  /  {bic['max']:.6f}",
        f"  Max |entry−0.5|   :  {bic['deviation_max']:.6f}",
        f"  Matrix shape      :  {m}×{m}  (diagonal = N/A)",
        _S("BIC-SAC Matrix  [row/col = output bit index]"),
        "  BIC[i][j] = avg_k Pr[ bit_i(Δ_k) XOR bit_j(Δ_k) = 1 ]   ideal = 0.5\n",
        _mat_txt(
            bic["matrix"],
            [f"Bit{i}" for i in range(m)],
            [f"Bit{j}" for j in range(m)],
        ),
    ])


def fmt_lat(r):
    lat = r["lat"]
    lines = [
        _H("LAT  —  LINEAR APPROXIMATION TABLE"),
        f"\n  Table shape          :  {lat['shape']}  (input masks × output masks)",
        f"  Max |LAT(a,b)|       :  {lat['max_abs']}  (a != 0, b != 0)",
        f"  Min nonzero |LAT|    :  {lat['min_nonzero_abs']}",
        f"  LP_max               :  {lat['lp_max']:.8f}",
        f"  NL from LAT          :  {lat['nl_from_lat']}",
        _S("Top 10 largest |LAT(a,b)| entries  (b ≠ 0)"),
        f"  {'a':>8}  {'b':>8}  {'LAT(a,b)':>12}",
        "  " + "─" * 34,
    ]
    for e in lat["top10"]:
        lines.append(f"  {e['a']:8d}  {e['b']:8d}  {e['value']:12d}")
    lines += [
        _S("Interpretation"),
        "  LAT(a,b) = Σ_x (−1)^{a·x XOR b·S(x)}.  Lower max → better.",
        "  AES S-box: max = 32, LP_max = 0.015625.",
        "  Full table is in the exported .txt report.",
    ]
    return "\n".join(lines)


def fmt_ddt(r):
    ddt = r["ddt"]
    N = len(r["ddt"]["table"])
    lines = [
        _H("DDT  —  DIFFERENCE DISTRIBUTION TABLE"),
        f"\n  Table shape                  :  {ddt['shape']}  (input diff × output diff)",
        f"  Differential Uniformity DU   :  {ddt['du']}",
        f"  Differential Probability DP  :  {ddt['dp']:.8f}  =  {ddt['du']}/{N}",
        f"  Min nonzero DDT count        :  {ddt['min_nonzero']}",
        _S("Top 10 largest DDT entries  (Δ_in ≠ 0)"),
        f"  {'Δ_in':>8}  {'Δ_out':>8}  {'Count':>8}",
        "  " + "─" * 32,
    ]
    for e in ddt["top10"]:
        lines.append(f"  {e['a']:8d}  {e['b']:8d}  {e['count']:8d}")
    lines += [
        _S("Interpretation"),
        "  DDT[Δin][Δout] = |{x : S(x XOR Δin) XOR S(x) = Δout}|.  Lower DU → better.",
        "  AES S-box: DU = 4, DP = 0.015625.",
        "  Full table is in the exported .txt report.",
    ]
    return "\n".join(lines)


def fmt_balancedness(r):
    bal = r["balancedness"]
    lines = [
        _H("BALANCEDNESS"),
        f"\n  Bijective (permutation)       :  {_B(bal['is_bijective'])}",
        f"  Distinct output values        :  {bal['distinct_outputs']}  / {bal['output_size']}",
        f"  Input size  N                 :  {bal['input_size']}",
        f"  Output space size  2^m        :  {bal['output_size']}",
        f"  Output frequency  min / max   :  {bal['min_freq']} / {bal['max_freq']}",
        f"  All coordinate fns balanced   :  {_B(bal['all_coords_balanced'])}",
        _S(f"Per-output-bit breakdown  ({len(bal['coord_info'])} output bits)"),
        f"  {'Bit':>5}   {'Ones':>6}   {'Zeros':>6}   {'Balanced':>10}",
        "  " + "─" * 36,
    ]
    for c in bal["coord_info"]:
        lines.append(
            f"  {c['bit']:>5}   {c['ones']:>6}   {c['zeros']:>6}   {'Yes' if c['balanced'] else 'No':>10}"
        )
    lines += [
        _S("Interpretation"),
        "  An S-box is bijective only when n=m and it maps each input to a unique output.",
        "  Non-bijective S-boxes (e.g., DES 6→4) are allowed and widely used.",
    ]
    return "\n".join(lines)


def fmt_linear_structure(r):
    ls = r["linear_structure"]
    lines = [
        _H("LINEAR STRUCTURE"),
        f"\n  Has any linear structure  :  {_B(ls['has_any_ls'])}",
        f"  Bits with LS              :  {ls['bits_with_ls']} / {ls['total_bits']}",
        _S(f"Per-output-bit analysis  (checking {ls['total_bits']} output coordinate bits)"),
    ]
    for bname, bd in ls["by_bit"].items():
        tag = "  <-- LS FOUND" if bd["has_ls"] else ""
        lines.append(f"  {bname}: {bd['count']} structure(s){tag}")
        for s in bd["structures"]:
            lines.append(
                f"      u = {s['u']:5d}  →  f_j(x XOR u) XOR f_j(x) = {s['constant']} for all x"
            )
    lines += [
        _S("Interpretation"),
        "  A non-zero linear structure is a cryptographic weakness.",
        "  A strong S-box has NO non-trivial linear structures.",
    ]
    return "\n".join(lines)


def fmt_autocorrelation(r):
    ac = r["autocorrelation"]
    lines = [
        _H("AUTOCORRELATION SPECTRUM"),
        f"\n  ABAC (max |AC_f(u)| for u≠0)  :  {ac['abac']}",
        _S(f"Per-output-bit autocorrelation  ({len(ac['by_bit'])} output bits)"),
        f"  {'Bit':>8}   {'max |AC(u≠0)|':>16}   {'AC(0)':>10}",
        "  " + "─" * 42,
    ]
    for bname, bd in ac["by_bit"].items():
        bit_idx = bname.split("_")[-1]
        lines.append(
            f"  {bit_idx:>8}   {bd['max_abs']:>16.4f}   {bd['ac0']:>10.2f}"
        )
    lines += [
        _S("Interpretation"),
        "  Lower ABAC → better.  AES S-box: ABAC = 32.",
        "  AC computed via Wiener-Khinchin: AC = WHT(W²) / N.",
    ]
    return "\n".join(lines)


def fmt_hybrid(r):
    hyb = r["hybrid_lat_ddt"]
    return "\n".join([
        _H("HYBRID LAT/DDT  —  BRIDGE IDENTITY"),
        f"\n  Vectorial NL       :  {hyb['nl_vectorial']}",
        f"  LAT max  M(S)      :  {hyb['lat_max']}",
        f"  LP_max             :  {hyb['lp_max']:.8f}",
        f"  DU                 :  {hyb['du']}",
        f"  DP_max             :  {hyb['dp']:.8f}",
        f"\n  {hyb['nl_formula']}",
        f"  {hyb['dp_formula']}",
        _S("Wiener-Khinchin Bridge Identity  (u = 1 spot test)"),
        "  For bijective S:  Σ_{b≠0} Σ_a  LAT(a,b)²  (−1)^{a·1}  =  −N²",
        f"  Result  :  {hyb['bridge_check']}",
        _S("Combined resistance"),
        f"  Linear  (LP_max)       :  {hyb['lp_max']:.6f}   (AES: 0.015625)",
        f"  Differential (DP_max)  :  {hyb['dp']:.6f}   (AES: 0.015625)",
    ])


def build_full_export(sbox, results, n_in, m_out):
    N  = len(sbox)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_size = 1 << m_out

    parts = [
        "=" * 70,
        "  S-Box Cryptographic Analyzer",
        "  Dr. Dawood Shah",
        "  Quaid-i-Azam University",
        "  Email: dawoodshah@math.qau.edu.pk",
        "=" * 70,
        "  S-BOX CRYPTOGRAPHIC ANALYSIS REPORT",
        f"  Generated  :  {ts}",
        f"  S-Box type :  {n_in}→{m_out}  ({n_in}-bit input, {m_out}-bit output)",
        f"  Entries    :  {N}",
        f"  Max output :  {int(max(sbox))}",
        "=" * 70,
        "\nS-Box values:\n" + ", ".join(str(v) for v in sbox),
        fmt_summary(results, sbox, n_in, m_out),
        fmt_comp_nl(results),
        fmt_sac(results),
        fmt_bic_sac(results),
        fmt_lat(results),
        fmt_ddt(results),
        fmt_balancedness(results),
        fmt_linear_structure(results),
        fmt_autocorrelation(results),
        fmt_hybrid(results),
        "\n" + "=" * 70,
        f"  FULL LAT TABLE  (rows = a  cols = b  value = LAT(a,b))",
        f"  Shape: {N}×{out_size}",
        "=" * 70,
    ]

    lat_t = results["lat"]["table"]
    hdr_lat = "    " + " ".join(f"{b:5d}" for b in range(out_size))
    parts.append(hdr_lat)
    for a, row in enumerate(lat_t):
        parts.append(f"{a:4d}" + " ".join(f"{v:5d}" for v in row))

    ddt_t = results["ddt"]["table"]
    parts += [
        "\n" + "=" * 70,
        f"  FULL DDT TABLE  (rows = Δ_in  cols = Δ_out  value = count)",
        f"  Shape: {N}×{out_size}",
        "=" * 70,
        hdr_lat,
    ]
    for di, row in enumerate(ddt_t):
        parts.append(f"{di:4d}" + " ".join(f"{v:5d}" for v in row))

    return "\n".join(parts)


# ===================================================================
#  SECTION 3 — QT GUI
# ===================================================================
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QTextEdit, QLabel, QPushButton, QFileDialog,
    QGroupBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QMessageBox, QGridLayout, QStackedWidget,
    QScrollArea, QSizePolicy,
)
from PyQt5.QtCore  import Qt, QThread, pyqtSignal
from PyQt5.QtGui   import QFont, QColor, QTextCursor, QIcon, QPainter, QLinearGradient

MONO = QFont("Consolas", 10)

APP_STYLE = """
QMainWindow {
    background-color: #020817;
}

QWidget {
    background-color: #020817;
    color: #FFFFFF;
    font-family: Segoe UI;
    font-size: 10.5pt;
}

QFrame#Sidebar {
    background-color: #03142C;
    border-right: 1px solid #164E85;
}

QFrame#Card, QFrame#HomePanel {
    background-color: #081F3F;
    border: 1px solid #164E85;
    border-radius: 14px;
}

QLabel#AppTitle {
    color: #FFFFFF;
    font-size: 24pt;
    font-weight: bold;
}

QLabel#SectionTitle {
    color: #FFFFFF;
    font-size: 16pt;
    font-weight: bold;
}

QLabel#MetricTitle {
    color: #B9D7FF;
    font-size: 10pt;
}

QLabel#MetricValue {
    color: #FFFFFF;
    font-size: 24pt;
    font-weight: bold;
}

QLabel#MetricHint {
    color: #8FBCEB;
    font-size: 9pt;
}

QLabel#AuthorLabel {
    color: #B9D7FF;
    font-size: 10.5pt;
}

QLabel#SubLabel, QLabel#hint, QLabel#seclbl, QLabel#infolbl {
    color: #B9D7FF;
}

QLabel#infoval {
    color: #FFFFFF;
}

QPushButton {
    background-color: #0B5ED7;
    color: #FFFFFF;
    border: 1px solid #168BFF;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #168BFF;
}

QPushButton:pressed {
    background-color: #064AAB;
}

QPushButton:disabled {
    background-color: #061A33;
    color: #6F8FB3;
    border-color: #164E85;
}

QPushButton#PrimaryButton, QPushButton#hero {
    background-color: #00AEEF;
    color: #001528;
    font-size: 12pt;
    padding: 13px 22px;
}

QPushButton#PrimaryButton:hover, QPushButton#hero:hover {
    background-color: #38C8FF;
}

QPushButton#danger {
    background-color: #03142C;
    color: #B9D7FF;
}

QPushButton#metric {
    background-color: #061A33;
    color: #B9D7FF;
    border: 1px solid #164E85;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 9.5pt;
    text-align: left;
}

QPushButton#metric:hover {
    background-color: #0A3A72;
}

QPushButton#metric:disabled {
    background-color: #020817;
    color: #3A567A;
    border-color: #0A1E35;
}

QPushButton#export {
    background-color: #0284C7;
    color: #FFFFFF;
}

QTabWidget::pane {
    background-color: #020817;
    border: 1px solid #164E85;
    border-radius: 8px;
}

QTabBar::tab {
    background-color: #03142C;
    color: #B9D7FF;
    padding: 11px 18px;
    border: 1px solid #164E85;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #0B5ED7;
    color: #FFFFFF;
    font-weight: bold;
}

QTabBar::tab:hover {
    background-color: #0A3A72;
}

QGroupBox {
    background-color: #081F3F;
    border: 1px solid #164E85;
    border-radius: 12px;
    color: #FFFFFF;
    font-weight: bold;
    margin-top: 10px;
    padding: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

QTableWidget {
    background-color: #061A33;
    alternate-background-color: #08264B;
    gridline-color: #164E85;
    color: #FFFFFF;
    border: 1px solid #164E85;
    border-radius: 8px;
    selection-background-color: #0B5ED7;
}

QHeaderView::section {
    background-color: #0A3A72;
    color: #FFFFFF;
    padding: 8px;
    border: 1px solid #164E85;
    font-weight: bold;
}

QTextEdit, QPlainTextEdit {
    background-color: #061A33;
    color: #EAF4FF;
    border: 1px solid #164E85;
    border-radius: 8px;
    padding: 10px;
    font-family: Consolas;
    font-size: 10pt;
}

QProgressBar {
    background-color: #061A33;
    border: 1px solid #164E85;
    border-radius: 8px;
    color: #FFFFFF;
    text-align: center;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #00AEEF;
    border-radius: 7px;
}

QScrollBar:vertical {
    background-color: #03142C;
    width: 12px;
}

QScrollBar::handle:vertical {
    background-color: #00AEEF;
    border-radius: 6px;
}

QScrollBar:horizontal {
    background-color: #03142C;
    height: 12px;
}

QScrollBar::handle:horizontal {
    background-color: #00AEEF;
    border-radius: 6px;
}

QStatusBar {
    background-color: #03142C;
    color: #B9D7FF;
    border-top: 1px solid #164E85;
}
"""


# ── Worker thread ─────────────────────────────────────────────────────
class Worker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, sbox):
        super().__init__()
        self.sbox = sbox

    def run(self):
        try:
            core    = SBoxCore(self.sbox)
            results = core.run_all(
                cb=lambda s, t, l: self.progress.emit(s, t, l)
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


# ── Colour-coded matrix widget ────────────────────────────────────────
def _color_table(mat, n_rows, n_cols,
                 row_prefix="In", col_prefix="Out") -> QTableWidget:
    tw = QTableWidget(n_rows, n_cols)
    tw.setFont(MONO)
    tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    tw.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
    tw.setHorizontalHeaderLabels([f"{col_prefix} {j}" for j in range(n_cols)])
    tw.setVerticalHeaderLabels([f"{row_prefix} {i}"  for i in range(n_rows)])
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i][j]
            if isinstance(v, float) and math.isnan(v):
                item = QTableWidgetItem("—")
                item.setBackground(QColor("#075985"))
            else:
                item = QTableWidgetItem(f"{v:.4f}")
                dev  = abs(v - 0.5)
                if   dev < 0.05:
                    item.setBackground(QColor("#0284C7"))
                    item.setForeground(QColor("#ffffff"))
                elif dev < 0.10:
                    item.setBackground(QColor("#0EA5E9"))
                    item.setForeground(QColor("#ffffff"))
                else:
                    item.setBackground(QColor("#075985"))
                    item.setForeground(QColor("#ffffff"))
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            tw.setItem(i, j, item)
    return tw


# ── NL component table widget ─────────────────────────────────────────
def _nl_table_widget(comp_table, m) -> QTableWidget:
    tw = QTableWidget(len(comp_table), 4)
    tw.setFont(MONO)
    tw.setHorizontalHeaderLabels(["Mask b", f"Binary ({m}-bit)", "Walsh Max", "NL"])
    tw.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    tw.verticalHeader().setVisible(False)

    for row_idx, e in enumerate(comp_table):
        nl_val = e["nl"]
        if nl_val >= 100:
            nl_color = QColor("#0284C7")
        elif nl_val >= 64:
            nl_color = QColor("#0EA5E9")
        else:
            nl_color = QColor("#075985")

        for col_idx, (val, fmt_str) in enumerate([
            (e["b"],         str(e["b"])),
            (e["b"],         e["binary"]),
            (e["walsh_max"], str(e["walsh_max"])),
            (nl_val,         str(nl_val)),
        ]):
            item = QTableWidgetItem(fmt_str)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if col_idx == 3:
                item.setBackground(nl_color)
                item.setForeground(QColor("#ffffff"))
            tw.setItem(row_idx, col_idx, item)

    return tw


def _style_table(tw: QTableWidget):
    tw.setAlternatingRowColors(True)
    tw.setFont(MONO)
    tw.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    tw.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
    tw.setShowGrid(True)
    tw.setWordWrap(False)
    tw.setSelectionBehavior(QTableWidget.SelectRows)
    tw.setEditTriggers(QTableWidget.NoEditTriggers)
    tw.horizontalHeader().setDefaultSectionSize(72)
    tw.verticalHeader().setDefaultSectionSize(30)
    for r in range(tw.rowCount()):
        tw.setRowHeight(r, 30)
    return tw


def _int_table_widget(table, row_prefix, col_prefix, max_rows=64, max_cols=32):
    rows = len(table)
    cols = len(table[0]) if rows else 0
    show_rows = min(rows, max_rows or rows)
    show_cols = min(cols, max_cols or cols)
    tw = QTableWidget(show_rows, show_cols)
    tw.setHorizontalHeaderLabels([f"{col_prefix}{j}" for j in range(show_cols)])
    tw.setVerticalHeaderLabels([f"{row_prefix}{i}" for i in range(show_rows)])
    for i in range(show_rows):
        for j in range(show_cols):
            item = QTableWidgetItem(str(table[i][j]))
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            tw.setItem(i, j, item)
    return _style_table(tw)


def _table_preview_note(name, table, max_rows=64, max_cols=32):
    rows = len(table)
    cols = len(table[0]) if rows else 0
    shown_rows = min(rows, max_rows)
    shown_cols = min(cols, max_cols)
    if shown_rows == rows and shown_cols == cols:
        return f"{name} table shown in full: {rows} x {cols}."
    return (
        f"{name} preview: showing {shown_rows} x {shown_cols} of {rows} x {cols}. "
        "Use Export Report for the complete table."
    )


def _balancedness_table_widget(bal):
    rows = bal["coord_info"]
    tw = QTableWidget(len(rows), 4)
    tw.setHorizontalHeaderLabels(["Output Bit", "Ones", "Zeros", "Balanced"])
    tw.verticalHeader().setVisible(False)
    for i, row in enumerate(rows):
        vals = [row["bit"], row["ones"], row["zeros"], "YES" if row["balanced"] else "NO"]
        for j, val in enumerate(vals):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(Qt.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if j == 3:
                item.setForeground(QColor("#22C55E" if row["balanced"] else "#EF4444"))
            tw.setItem(i, j, item)
    return _style_table(tw)


def create_metric_card(title, value, hint="", status=None):
    card = QFrame()
    card.setObjectName("Card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(6)
    title_label = QLabel(title)
    title_label.setObjectName("MetricTitle")
    value_label = QLabel(str(value))
    value_label.setObjectName("MetricValue")
    hint_label = QLabel(hint)
    hint_label.setObjectName("MetricHint")
    hint_label.setWordWrap(True)
    layout.addWidget(title_label)
    layout.addWidget(value_label)
    layout.addWidget(hint_label)
    if status:
        badge = QLabel(status)
        up = status.upper()
        if up in ("PASS", "YES"):
            badge.setStyleSheet("color: #22C55E; font-weight: bold;")
        elif up in ("FAIL", "NO"):
            badge.setStyleSheet("color: #EF4444; font-weight: bold;")
        else:
            badge.setStyleSheet("color: #F59E0B; font-weight: bold;")
        layout.addWidget(badge)
    return card


def _summary_dashboard(results, sbox, n_in, m_out):
    nl = results["nonlinearity"]
    lat = results["lat"]
    ddt = results["ddt"]
    sac = results["sac"]
    bic = results["bic_sac"]
    bal = results["balancedness"]
    wrapper = QWidget()
    outer = QVBoxLayout(wrapper)
    outer.setContentsMargins(18, 18, 18, 18)
    outer.setSpacing(18)
    heading = QLabel(f"{n_in}→{m_out} S-box Dashboard")
    heading.setObjectName("SectionTitle")
    outer.addWidget(heading)
    sub = QLabel(
        f"{len(sbox)} entries  |  Output range 0 to {int(max(sbox))}  |  "
        f"{'Bijective permutation' if bal['is_bijective'] else 'Non-bijective'}"
    )
    sub.setObjectName("AuthorLabel")
    outer.addWidget(sub)
    grid = QGridLayout()
    grid.setSpacing(14)
    cards = [
        ("Minimum NL",      nl["nl_min"],           "Vectorial NL over all component masks", None),
        ("Maximum NL",      nl["nl_max"],            "Best component nonlinearity observed",  None),
        ("Average NL",      nl["nl_avg"],            "Mean over all nonzero output masks",    None),
        ("LAT max",         lat["max_abs"],          "Maximum nontrivial linear bias",        None),
        ("LPmax",           f"{lat['lp_max']:.8f}",  "Maximum linear probability",            None),
        ("Differential DU", ddt["du"],               "Maximum DDT count for dx != 0",         None),
        ("DPmax",           f"{ddt['dp']:.8f}",      "Maximum differential probability",      None),
        ("SAC Average",     f"{sac['avg']:.6f}",     "Average avalanche probability",
            "PASS" if sac["passes_criterion"] else "FAIL"),
        ("BIC-SAC Average", f"{bic['avg']:.6f}",    "Output bit-pair independence under SAC",
            "PASS" if bic["passes_criterion"] else "FAIL"),
        ("Balanced Bits",   "YES" if bal["all_coords_balanced"] else "NO",
            "Coordinate Boolean functions balanced",
            "YES" if bal["all_coords_balanced"] else "NO"),
        ("Bijective",       "YES" if bal["is_bijective"] else "NO",
            "Permutation property, only n=m",
            "YES" if bal["is_bijective"] else "NO"),
        ("Components",      nl["num_components"],    f"= 2^{m_out}-1 output masks analyzed",  None),
    ]
    for i, card in enumerate(cards):
        grid.addWidget(create_metric_card(*card), i // 4, i % 4)
    outer.addLayout(grid)
    outer.addStretch()
    return wrapper


# ── Result pane (text + optional extra widget) ────────────────────────
class ResultPane(QWidget):
    def __init__(self):
        super().__init__()
        self._lyt = QVBoxLayout(self)
        self._lyt.setContentsMargins(4, 4, 4, 4)
        self._lyt.setSpacing(6)

        self.te = QTextEdit()
        self.te.setFont(MONO)
        self.te.setReadOnly(True)
        self._lyt.addWidget(self.te)

        self._extra = None

    def set_text(self, text: str):
        self.te.setVisible(True)
        if self._extra is not None:
            self._lyt.removeWidget(self._extra)
            self._extra.deleteLater()
            self._extra = None
        self.te.setPlainText(text)
        self.te.moveCursor(QTextCursor.Start)

    def set_widget(self, widget):
        if self._extra is not None:
            self._lyt.removeWidget(self._extra)
            self._extra.deleteLater()
        self._extra = widget
        self._extra.setMaximumHeight(340)
        self._lyt.addWidget(self._extra)

    def set_widget_only(self, widget):
        self.te.clear()
        self.te.setVisible(False)
        if self._extra is not None:
            self._lyt.removeWidget(self._extra)
            self._extra.deleteLater()
        self._extra = widget
        self._extra.setMaximumHeight(16777215)
        self._lyt.addWidget(self._extra)


# ===================================================================
#  FRONT PAGE
# ===================================================================
class BinaryBackgroundWidget(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#031B3D"))
        gradient.setColorAt(1, QColor("#020817"))
        painter.fillRect(self.rect(), gradient)
        painter.setPen(QColor(120, 190, 255, 25))
        painter.setFont(QFont("Consolas", 13))
        pattern = "0101010101 1010101010 0101010101 1010101010"
        y = 35
        while y < self.height():
            x = 20
            while x < self.width():
                painter.drawText(x, y, pattern)
                x += 420
            y += 42


class FrontPage(QWidget):
    import_file  = pyqtSignal()
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")
        scroll.viewport().setStyleSheet("background: transparent;")
        outer.addWidget(scroll)

        container = BinaryBackgroundWidget()
        scroll.setWidget(container)

        root = QVBoxLayout(container)
        root.setAlignment(Qt.AlignHCenter | Qt.AlignCenter)
        root.setContentsMargins(70, 60, 70, 60)
        root.setSpacing(16)

        # ── Title ──────────────────────────────────────────────────────
        title = QLabel("S-Box Cryptographic Analyzer")
        title.setObjectName("AppTitle")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        tagline = QLabel(
            "Advanced cryptographic analysis for n→m S-boxes\n"
            "Square, non-square  ·  Bijective and non-bijective  ·  3-bit to 8-bit input"
        )
        tagline.setObjectName("AuthorLabel")
        tagline.setAlignment(Qt.AlignCenter)
        root.addWidget(tagline)

        root.addSpacing(10)

        # ── Import button ──────────────────────────────────────────────
        import_btn = QPushButton("IMPORT S-BOX")
        import_btn.setObjectName("PrimaryButton")
        import_btn.setFixedSize(380, 72)
        import_btn.clicked.connect(self.import_file)
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignCenter)
        btn_row.addWidget(import_btn)
        root.addLayout(btn_row)

        # ── Drag & drop hint ───────────────────────────────────────────
        drop_hint = QLabel("— or drag and drop a .txt / .csv file anywhere on this window —")
        drop_hint.setObjectName("SubLabel")
        drop_hint.setAlignment(Qt.AlignCenter)
        root.addWidget(drop_hint)

        root.addSpacing(6)

        # ── Supported formats panel ────────────────────────────────────
        panel = QFrame()
        panel.setObjectName("HomePanel")
        panel.setMaximumWidth(800)
        panel_lyt = QVBoxLayout(panel)
        panel_lyt.setContentsMargins(40, 28, 40, 28)
        panel_lyt.setSpacing(12)

        supported_title = QLabel("Supported S-box types")
        supported_title.setObjectName("SectionTitle")
        supported_title.setStyleSheet("font-size: 13pt;")
        supported_title.setAlignment(Qt.AlignCenter)
        panel_lyt.addWidget(supported_title)

        examples = QLabel(
            "AES 8→8: 256 entries, max value 255\n"
            "DES 6→4: 64 entries, max value 15\n"
            "4-bit S-box: 16 entries, max value 15\n"
            "3-bit S-box: 8 entries, max value 7\n"
            "Non-bijective 8→4: 256 entries, max value 15"
        )
        examples.setObjectName("SubLabel")
        examples.setAlignment(Qt.AlignCenter)
        panel_lyt.addWidget(examples)

        fmt_hint = QLabel(
            "Accepted formats:  {0,1,2,...}   [0,1,2,...]   0x63,0x7c,...\n"
            "Decimal or hexadecimal — comma, space, tab, or newline separated"
        )
        fmt_hint.setObjectName("SubLabel")
        fmt_hint.setAlignment(Qt.AlignCenter)
        panel_lyt.addWidget(fmt_hint)

        panel_row = QHBoxLayout()
        panel_row.setAlignment(Qt.AlignCenter)
        panel_row.addWidget(panel)
        root.addLayout(panel_row)

        root.addSpacing(10)

        # ── Metrics footer ─────────────────────────────────────────────
        metrics_lbl = QLabel(
            "Computes: Nonlinearity · LAT · DDT · SAC · BIC-SAC · "
            "Balancedness · Linear Structure · Autocorrelation · Hybrid LAT/DDT"
        )
        metrics_lbl.setObjectName("SubLabel")
        metrics_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(metrics_lbl)

        root.addSpacing(10)

        # ── Author ─────────────────────────────────────────────────────
        author = QLabel(
            "Dr. Dawood Shah  ·  Quaid-i-Azam University  ·  dawoodshah@math.qau.edu.pk"
        )
        author.setObjectName("AuthorLabel")
        author.setAlignment(Qt.AlignCenter)
        root.addWidget(author)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                self.file_dropped.emit(path)
                return


# ===================================================================
#  ANALYSIS PANEL
# ===================================================================
class AnalysisPanel(QWidget):
    back_clicked   = pyqtSignal()
    import_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._sbox    = None
        self._n_in    = None
        self._m_out   = None
        self._results = None
        self._worker  = None
        self._build()

    def _build(self):
        root_lyt = QHBoxLayout(self)
        root_lyt.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        root_lyt.addWidget(splitter)

        # ── LEFT SIDEBAR ─────────────────────────────────────────────
        left = QFrame()
        left.setObjectName("Sidebar")
        left.setFixedWidth(230)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("border: none; background: transparent;")
        scroll_area.viewport().setStyleSheet("background: transparent;")

        inner = QWidget()
        inner.setObjectName("Sidebar")
        lv = QVBoxLayout(inner)
        lv.setContentsMargins(14, 14, 10, 14)
        lv.setSpacing(8)

        scroll_area.setWidget(inner)

        sidebar_lyt = QVBoxLayout(left)
        sidebar_lyt.setContentsMargins(0, 0, 0, 0)
        sidebar_lyt.addWidget(scroll_area)

        hdr = QLabel("S-Box Analyzer")
        hdr.setObjectName("SectionTitle")
        hdr.setStyleSheet("font-size: 13pt;")
        hdr.setAlignment(Qt.AlignCenter)
        lv.addWidget(hdr)

        author = QLabel("Dr. Dawood Shah\nQuaid-i-Azam University")
        author.setObjectName("AuthorLabel")
        author.setStyleSheet("font-size: 9pt;")
        author.setAlignment(Qt.AlignCenter)
        lv.addWidget(author)

        # Info group
        grp = QGroupBox("Current S-Box")
        gi  = QGridLayout(grp)
        gi.setSpacing(4)
        gi.setColumnStretch(1, 1)

        def _inforow(label):
            lbl = QLabel(label + ":")
            lbl.setObjectName("infolbl")
            lbl.setStyleSheet("font-size: 9pt;")
            val = QLabel("—")
            val.setObjectName("infoval")
            val.setStyleSheet("font-size: 9pt;")
            val.setWordWrap(True)
            return lbl, val

        ll_type,  self._lbl_type   = _inforow("Type")
        ll_ibits, self._lbl_ibits  = _inforow("Input bits")
        ll_obits, self._lbl_obits  = _inforow("Output bits")
        ll_ent,   self._lbl_ent    = _inforow("Entries")
        ll_max,   self._lbl_max    = _inforow("Output range")
        ll_bij,   self._lbl_bij    = _inforow("Bijective")

        for r, (l, v) in enumerate([
            (ll_type,  self._lbl_type),
            (ll_ibits, self._lbl_ibits),
            (ll_obits, self._lbl_obits),
            (ll_ent,   self._lbl_ent),
            (ll_max,   self._lbl_max),
            (ll_bij,   self._lbl_bij),
        ]):
            gi.addWidget(l, r, 0)
            gi.addWidget(v, r, 1)
        lv.addWidget(grp)

        # Primary action buttons
        self._btn_new = QPushButton("Import S-Box")
        self._btn_new.setObjectName("PrimaryButton")
        self._btn_new.setFixedHeight(48)
        self._btn_new.clicked.connect(self.import_clicked)
        lv.addWidget(self._btn_new)

        self._btn_run = QPushButton("Run Full Analysis")
        self._btn_run.setFixedHeight(38)
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._run_all)
        lv.addWidget(self._btn_run)

        self._progress = QProgressBar()
        self._progress.setRange(0, 9)
        self._progress.setVisible(False)
        lv.addWidget(self._progress)

        # Individual metric buttons — added to layout and enabled on sbox load
        sec = QLabel("Individual metrics:")
        sec.setObjectName("seclbl")
        sec.setStyleSheet("font-size: 9pt; padding-top: 4px;")
        lv.addWidget(sec)

        self._metric_btns = {}
        METRICS = [
            ("Nonlinearity",     "nonlinearity"),
            ("SAC",              "sac"),
            ("BIC-SAC",          "bic_sac"),
            ("LAT",              "lat"),
            ("DDT",              "ddt"),
            ("Balancedness",     "balancedness"),
            ("Linear Structure", "linear_structure"),
            ("Autocorrelation",  "autocorrelation"),
            ("Hybrid LAT/DDT",   "hybrid_lat_ddt"),
        ]
        for label, key in METRICS:
            btn = QPushButton(label)
            btn.setObjectName("metric")
            btn.setFixedHeight(28)
            btn.setEnabled(False)
            btn.clicked.connect(
                lambda _, k=key, lbl=label: self._run_single(k, lbl)
            )
            self._metric_btns[key] = btn
            lv.addWidget(btn)

        lv.addStretch()

        self._btn_export = QPushButton("Export Report")
        self._btn_export.setObjectName("export")
        self._btn_export.setFixedHeight(38)
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export)
        lv.addWidget(self._btn_export)

        self._btn_back = QPushButton("Home")
        self._btn_back.setObjectName("danger")
        self._btn_back.setFixedHeight(28)
        self._btn_back.clicked.connect(self.back_clicked)
        lv.addWidget(self._btn_back)

        splitter.addWidget(left)

        # ── RIGHT PANEL (tabs) ────────────────────────────────────────
        self._tabs = QTabWidget()
        TAB_DEFS = [
            ("summary",          "Summary"),
            ("comp_nl",          "Component NL"),
            ("lat",              "LAT"),
            ("ddt",              "DDT"),
            ("sac",              "SAC"),
            ("bic_sac",          "BIC-SAC"),
            ("balancedness",     "Balancedness"),
            ("advanced",         "Advanced"),
            ("full_report",      "Report"),
        ]
        self._panes    = {}
        self._tab_keys = []
        for key, label in TAB_DEFS:
            pane = ResultPane()
            pane.set_text(
                "S-box loaded.\n"
                "Click 'Run Full Analysis' to compute all metrics,\n"
                "or click an individual metric button on the left."
            )
            self._panes[key]    = pane
            self._tab_keys.append(key)
            self._tabs.addTab(pane, label)

        splitter.addWidget(self._tabs)
        splitter.setSizes([230, 1060])

    # ── Load S-box ────────────────────────────────────────────────────
    def load_sbox(self, sbox: list, source: str = ""):
        try:
            core = SBoxCore(sbox)
        except ValueError as e:
            QMessageBox.critical(self, "Invalid S-Box", str(e))
            return False

        self._sbox    = sbox
        self._n_in    = core.n
        self._m_out   = core.m
        self._results = None

        counts   = Counter(sbox)
        is_bij   = (core.n == core.m) and (len(counts) == len(sbox)) and all(v == 1 for v in counts.values())

        self._lbl_type.setText(f"{core.n}→{core.m} S-box")
        self._lbl_ibits.setText(str(core.n))
        self._lbl_obits.setText(str(core.m))
        self._lbl_ent.setText(str(core.N))
        self._lbl_max.setText(f"0 to {int(max(sbox))}")
        self._lbl_bij.setText("Yes" if is_bij else "No")
        self._lbl_bij.setStyleSheet(
            "color: #22C55E; font-size: 9pt;" if is_bij else "color: #EF4444; font-size: 9pt;"
        )

        self._btn_run.setEnabled(True)
        self._btn_export.setEnabled(False)
        for b in self._metric_btns.values():
            b.setEnabled(True)

        for pane in self._panes.values():
            pane.set_text(
                f"S-box loaded: {len(sbox)} entries ({core.n}→{core.m}) from '{source}'.\n\n"
                "Click  'Run Full Analysis'  to compute all 9 metrics,\n"
                "or click an individual metric button in the left panel."
            )

        self._run_all()
        return True

    # ── Run all analyses ──────────────────────────────────────────────
    def _run_all(self):
        if not self._sbox:
            return
        self._set_busy(True)
        self._worker = Worker(self._sbox)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ── Run single metric ─────────────────────────────────────────────
    def _run_single(self, key, label):
        if not self._sbox:
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()

            core   = SBoxCore(self._sbox)
            result = getattr(core, key)()

            QApplication.restoreOverrideCursor()

            fmt_map = {
                "nonlinearity":     lambda: fmt_comp_nl({"nonlinearity": result}),
                "sac":              lambda: fmt_sac({"sac": result}),
                "bic_sac":          lambda: fmt_bic_sac({"bic_sac": result}),
                "lat":              lambda: fmt_lat({"lat": result}),
                "ddt":              lambda: fmt_ddt({"ddt": result}),
                "balancedness":     lambda: fmt_balancedness({"balancedness": result}),
                "linear_structure": lambda: fmt_linear_structure({"linear_structure": result}),
                "autocorrelation":  lambda: fmt_autocorrelation({"autocorrelation": result}),
                "hybrid_lat_ddt":   lambda: fmt_hybrid({"hybrid_lat_ddt": result}),
            }

            pane_key = "comp_nl" if key == "nonlinearity" else key
            if pane_key in self._panes:
                self._panes[pane_key].set_text(fmt_map[key]())

                if key == "nonlinearity":
                    self._panes[pane_key].set_widget(
                        _nl_table_widget(result["comp_table"], core.m)
                    )
                elif key == "sac":
                    self._panes[pane_key].set_widget(
                        _color_table(result["matrix"], result["n_rows"],
                                     result["n_cols"], "In", "Out")
                    )
                elif key == "bic_sac":
                    self._panes[pane_key].set_widget(
                        _color_table(result["matrix"], result["n_size"],
                                     result["n_size"], "Bit", "Bit")
                    )

                idx = self._tab_keys.index(pane_key)
                self._tabs.setCurrentIndex(idx)

        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", str(e))

    # ── Export ────────────────────────────────────────────────────────
    def _export(self):
        if not self._results:
            QMessageBox.warning(
                self, "No Full Results",
                "Please click 'Run Full Analysis' first to generate the full report."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "sbox_analysis_results.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            content = build_full_export(
                self._sbox, self._results, self._n_in, self._m_out
            )
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            QMessageBox.information(
                self, "Exported",
                f"Results saved to:\n{path}\n\n"
                "Includes full component NL table, full LAT and DDT tables."
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ── Worker callbacks ──────────────────────────────────────────────
    def _set_busy(self, busy: bool):
        self._btn_run.setEnabled(not busy)
        self._btn_new.setEnabled(not busy)
        for b in self._metric_btns.values():
            b.setEnabled(not busy)
        self._progress.setVisible(busy)
        if busy:
            self._progress.setValue(0)
            self._progress.setFormat("Starting…")

    def _on_progress(self, step: int, total: int, label: str):
        self._progress.setValue(step)
        self._progress.setFormat(f"{label}…  ({step}/{total})")

    def _on_finished(self, results: dict):
        self._results = results
        self._set_busy(False)
        self._btn_export.setEnabled(True)

        sbox  = self._sbox
        n_in  = self._n_in
        m_out = self._m_out

        nl_r  = results["nonlinearity"]
        sac_r = results["sac"]
        bic_r = results["bic_sac"]

        # Summary dashboard
        self._panes["summary"].set_widget_only(
            _summary_dashboard(results, sbox, n_in, m_out)
        )

        # Component NL table
        self._panes["comp_nl"].set_text(
            f"Component nonlinearity for all {nl_r['num_components']} nonzero output masks (b = 1 … 2^m-1)."
        )
        self._panes["comp_nl"].set_widget_only(
            _nl_table_widget(nl_r["comp_table"], m_out)
        )

        # LAT
        self._panes["lat"].set_text(
            fmt_lat(results) + "\n\n  " +
            _table_preview_note("LAT", results["lat"]["table"])
        )
        self._panes["lat"].set_widget(
            _int_table_widget(results["lat"]["table"], "a=", "b=")
        )

        # DDT
        self._panes["ddt"].set_text(
            fmt_ddt(results) + "\n\n  " +
            _table_preview_note("DDT", results["ddt"]["table"])
        )
        self._panes["ddt"].set_widget(
            _int_table_widget(results["ddt"]["table"], "dx=", "dy=")
        )

        # SAC
        self._panes["sac"].set_text(fmt_sac(results))
        self._panes["sac"].set_widget_only(
            _color_table(sac_r["matrix"], sac_r["n_rows"],
                         sac_r["n_cols"], "In", "Out")
        )

        # BIC-SAC
        self._panes["bic_sac"].set_text(fmt_bic_sac(results))
        self._panes["bic_sac"].set_widget_only(
            _color_table(bic_r["matrix"], bic_r["n_size"],
                         bic_r["n_size"], "Bit", "Bit")
        )

        # Balancedness
        self._panes["balancedness"].set_text(fmt_balancedness(results))
        self._panes["balancedness"].set_widget(
            _balancedness_table_widget(results["balancedness"])
        )

        # Advanced
        self._panes["advanced"].set_text(
            fmt_autocorrelation(results) + "\n\n" +
            fmt_linear_structure(results) + "\n\n" +
            fmt_hybrid(results)
        )

        # Full Report tab
        self._panes["full_report"].set_text(
            "Full report is ready to export.\n\n"
            "Click 'Export Report' in the sidebar to generate the complete .txt file.\n\n"
            "The report includes:\n"
            f"  • All {nl_r['num_components']} component nonlinearities (b = 1 … 2^m-1)\n"
            f"  • Full {results['lat']['shape']} LAT table\n"
            f"  • Full {results['ddt']['shape']} DDT table\n"
            f"  • {sac_r['n_rows']}×{sac_r['n_cols']} SAC matrix\n"
            f"  • {bic_r['n_size']}×{bic_r['n_size']} BIC-SAC matrix\n"
            "  • Balancedness, Linear Structure, Autocorrelation\n\n"
            "This tab avoids rendering all tables directly so large S-boxes "
            "do not freeze the interface."
        )

        self._tabs.setCurrentIndex(0)

    def _on_error(self, msg: str):
        self._set_busy(False)
        QMessageBox.critical(self, "Analysis Error", msg)


# ===================================================================
#  MAIN WINDOW
# ===================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S-Box Cryptographic Analyzer")
        self.setMinimumSize(1100, 680)
        self.resize(1380, 860)
        self.setStyleSheet(APP_STYLE)
        self.setAcceptDrops(True)
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self._stack    = QStackedWidget()
        self._front    = FrontPage()
        self._analysis = AnalysisPanel()

        self._stack.addWidget(self._front)    # index 0
        self._stack.addWidget(self._analysis) # index 1
        self.setCentralWidget(self._stack)

        self._front.import_file.connect(self._import_file)
        self._front.file_dropped.connect(self._load_file_path)
        self._analysis.import_clicked.connect(self._import_file)
        self._analysis.back_clicked.connect(self._show_front)

        self.statusBar().showMessage(
            "Welcome — import or drag-and-drop an S-box .txt / .csv file to begin."
        )

    def _show_front(self):
        self._stack.setCurrentIndex(0)

    def _import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import S-Box", "",
            "Text/CSV Files (*.txt *.csv);;All Files (*)"
        )
        if path:
            self._load_file_path(path)

    def _load_file_path(self, path: str):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            sbox = parse_sbox(text)
            ok   = self._analysis.load_sbox(sbox, os.path.basename(path))
            if ok:
                self._stack.setCurrentIndex(1)
                self.statusBar().showMessage(
                    f"Loaded {len(sbox)} entries from {os.path.basename(path)} — "
                    "running all analyses…"
                )
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                self._load_file_path(path)
                return


# ===================================================================
#  ENTRY POINT
# ===================================================================
def main():
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps,    True)
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("S-Box Cryptographic Analyzer")
    app.setStyleSheet(APP_STYLE)
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
