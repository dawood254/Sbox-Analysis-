============================================================
  S-Box Cryptographic Analyzer  —  User Guide
============================================================

WHAT THIS SOFTWARE COMPUTES
-----------------------------
For any 4-bit (16-entry) or 8-bit (256-entry) S-box it computes:

  1. Nonlinearity       — Vectorial NL (all 2^n−1 component fns),
                          Coordinate NL (unit masks), LAT max M(S),
                          Gap between coordinate and vectorial NL.

  2. SAC                — Strict Avalanche Criterion 8×8 matrix.
                          Colour-coded: green=good, orange=borderline,
                          red=poor.

  3. BIC-SAC            — Bit Independence Criterion (SAC variant):
                          independence of output bit pairs under single-
                          bit input flips.

  4. LAT                — Full Linear Approximation Table (256×256 for
                          8-bit S-boxes).  Shows top-10 largest entries
                          and LP_max; full table saved to export file.

  5. DDT                — Full Difference Distribution Table.
                          Reports differential uniformity DU and DP_max.

  6. Balancedness       — Bijectivity check and per-coordinate balance.

  7. Linear Structure   — Per-coordinate function search for non-trivial
                          linear structures (u: f(x⊕u)⊕f(x)=const).

  8. Autocorrelation    — Per-coordinate ABAC (Absolute Autocorrelation)
                          via Wiener-Khinchin identity.

  9. Hybrid LAT/DDT     — Bridge identity verification:
                          sum_{b≠0} sum_a LAT(a,b)² (−1)^{a·u} = −N²


HOW TO USE
-----------
Method A — Run from Python (development / testing):
  1. Open a terminal in this folder.
  2. pip install PyQt5 numpy
  3. python main.py

Method B — Build a standalone Windows .exe:
  1. Double-click  build_exe.bat
  2. Wait 1-2 minutes for PyInstaller to finish.
  3. The .exe appears in:  dist\SBoxAnalyzer.exe
  4. Copy SBoxAnalyzer.exe anywhere; no Python required to run it.


INPUT FILE FORMAT
------------------
Create a plain .txt file containing the 256 S-box values in any of
these formats — the importer handles all of them:

  {99, 124, 119, 123, 242, 107, 111, ...}   ← braces, commas
  [99, 124, 119, 123, 242, 107, 111, ...]   ← brackets
  99 124 119 123 242 107 111 ...             ← space-separated
  0x63, 0x7c, 0x77, 0x7b, ...               ← hexadecimal

Values must be in [0, 255] for 8-bit, [0, 15] for 4-bit S-boxes.


EXPORTING RESULTS
------------------
After running the analysis, click "Export Results to .txt".
The exported file contains:
  • All nine analyses with interpretation notes
  • Complete 256×256 LAT table
  • Complete 256×256 DDT table


REFERENCE S-BOXES FOR TESTING
-------------------------------
AES SubBytes (first 16 values):
  {99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,...}

Full AES S-box can be pasted from Wikipedia:
  https://en.wikipedia.org/wiki/Rijndael_S-box


CRYPTOGRAPHIC NOTES
--------------------
For an 8-bit bijective S-box, reference values (AES SubBytes):
  NL_vec       = 112
  M(S)         = 32
  DU           = 4
  DP_max       = 0.015625
  SAC avg      ≈ 0.5
  Linear struct: none

============================================================
