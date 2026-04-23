# PQC Side-Channel Scanner

An open-source automated timing side-channel scanner for NIST post-quantum cryptography standards. Detects constant-time violations in ML-KEM (FIPS 203) and ML-DSA (FIPS 204) implementations using TVLA statistical methodology.

Built by [Collective Qubits](https://github.com/Disha231102004).

---

## Why This Exists

NIST finalized the post-quantum cryptography standards in August 2024. Banks, cloud providers, and government agencies are actively deploying ML-KEM and ML-DSA right now to protect against future quantum attacks. But timing side-channel vulnerabilities in cryptographic implementations can leak secret key material to an attacker — even when the algorithm itself is mathematically secure.

Existing tools like `dudect` require manual C integration, have no PQC-specific targets, and predate the NIST standards by years. No open-source automated scanner existed for FIPS 203 and FIPS 204 specifically.

This tool fills that gap.

---

## What It Found

Running against liboqs — the production PQC library used by AWS, Cloudflare, and Microsoft:

| Operation | Delta | t-statistic | p-value | Verdict |
|---|---|---|---|---|
| ML-KEM Encapsulation | 264ns | 75.5 | p=0.0 | **CRITICAL LEAK** |
| ML-KEM Decapsulation | 37ns | -12.4 | p=2.21e-35 | **LEAK** |

C-level noise floor (AES-128): 11.7ns. The encapsulation finding is 22x above the noise floor.

---

## How It Works

The scanner implements **Test Vector Leakage Assessment (TVLA)** — the same methodology used by Riscure and documented in ISO 17825. For each cryptographic operation it:

1. Collects timing measurements for a **fixed input** (same value every time)
2. Collects timing measurements for **random inputs** (different every time)
3. Runs **Welch's t-test** on the two distributions
4. If `|t| > 4.5` the implementation is leaking — its timing depends on the input, which means an attacker can learn information about secret keys

The scanner has two measurement layers:
- **Python layer** — targets `kyber-py` and `dilithium-py`, detects millisecond-level leaks
- **C layer** — targets `liboqs` directly with `CLOCK_MONOTONIC_RAW`, detects nanosecond-level leaks

---

## Installation

### Requirements

- Ubuntu 20.04+ or WSL2 (Ubuntu 24.04 tested)
- Python 3.10+
- gcc and clang
- liboqs (instructions below)

### 1. Clone the repository

```bash
git clone https://github.com/Disha231102004/pqc-scanner
cd pqc-scanner
```

### 2. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install kyber-py dilithium-py scipy numpy click rich
```

### 3. Install liboqs

```bash
sudo apt install -y cmake ninja-build libssl-dev clang

git clone --depth=1 https://github.com/open-quantum-safe/liboqs ~/liboqs
cmake -S ~/liboqs -B ~/liboqs/build -DBUILD_SHARED_LIBS=ON -GNinja
cmake --build ~/liboqs/build --parallel 4
sudo cmake --build ~/liboqs/build --target install
sudo ldconfig
```

### 4. Build the C harnesses

```bash
cd harness

# Production scanner (liboqs ML-KEM-768)
gcc -O2 -o ml_kem_harness ml_kem_harness.c -loqs -lssl -lcrypto

# Noise floor baseline (AES-128)
gcc -O2 -o baseline_harness baseline_harness.c -lssl -lcrypto

cd ..
```

---

## Usage

Always activate the virtual environment first:

```bash
source venv/bin/activate
```

### Scan ML-KEM and ML-DSA (Python targets)

```bash
python cli.py scan --algorithm both --traces 10000 --pin-cpu
```

Options:
- `--algorithm` — `ml-kem`, `ml-dsa`, or `both`
- `--traces` — number of measurements per operation (default 10000)
- `--pin-cpu` — pin process to CPU core 0 for lower noise
- `--no-exit` — do not exit with code 1 on findings (for scripting)
- `--open-html` — open HTML report in browser automatically

### Production scan (liboqs C library)

```bash
python cli.py liboqs --traces 100000
```

This is the primary scan. Uses `CLOCK_MONOTONIC_RAW` via C harness for nanosecond precision. 100,000 traces recommended for statistical confidence.

### Algorithm comparison

```bash
python cli.py compare --traces 10000 --pin-cpu
```

Scans ML-KEM-768 and ML-DSA-65 side by side. Produces a comparison HTML report with bar chart.

### Python noise floor characterization

```bash
python cli.py baseline --traces 10000 --pin-cpu
```

Runs SHA-256 as a control experiment. SHA-256 is provably constant-time so any signal here is measurement noise. Run this before reporting findings.

### C-level noise floor characterization

```bash
python cli.py c-baseline --traces 100000
```

Runs AES-128 (AES-NI hardware instruction) through the same C harness. Reports the noise floor delta so you can assess the signal-to-noise ratio of liboqs findings.

### Compiler flag sweep

```bash
python cli.py compiler-sweep --traces 50000 --runs 3
```

Builds the C harness with six compiler flag combinations and scans each. Looking for optimization-induced timing leaks — the same search space where KyberSlash was found. Each config is run `--runs` times and the median t-statistic reported for stability.

---

## Output

Every scan produces two files:

- `*.json` — structured findings for CI/CD integration
- `*.html` — interactive report with timing distribution charts

### JSON format

```json
{
  "scanner": "pqc-scanner v0.3",
  "target": "liboqs ML-KEM-768",
  "summary": {
    "total": 1,
    "critical": 1,
    "high": 0,
    "medium": 0
  },
  "findings": [
    {
      "operation": "ML-KEM Encapsulation",
      "t_statistic": 75.5185,
      "p_value": 0.0,
      "delta_ns": 264.0,
      "severity": "CRITICAL",
      "verdict": "LEAK DETECTED"
    }
  ]
}
```

### CI/CD integration

The scanner exits with code `1` when findings are detected, making it compatible with any CI pipeline:

```yaml
# GitHub Actions example
- name: PQC Timing Scan
  run: |
    source venv/bin/activate
    python cli.py liboqs --traces 50000
  # Pipeline fails automatically if leaks are found
```

Use `--no-exit` to suppress the exit code when running in reporting-only mode.

---

## Methodology

### TVLA (Test Vector Leakage Assessment)

Developed by Cryptography Research Inc. and documented in ISO 17825. The core principle: if an implementation is constant-time, its execution time should be statistically independent of the input. We test this by comparing timing distributions for fixed vs random inputs.

### Welch's t-test

Used instead of Student's t-test because it does not assume equal variance between the two distributions. Threshold: `|t| > 4.5` indicates a statistically significant timing difference, regardless of noise floor.

### Measurement precision

- **Python layer** — `time.perf_counter_ns()`, noise floor ~100-130ns in WSL
- **C layer** — `CLOCK_MONOTONIC_RAW` via `clock_gettime()`, noise floor ~10-12ns in WSL

### Noise reduction

- **500-trace warmup** — discards initial measurements to stabilize CPU cache and branch predictor
- **5% outlier trimming** — removes top 5% of measurements to eliminate OS interrupt spikes
- **CPU affinity** — `sched_setaffinity` pins the process to core 0 to reduce scheduler noise
- **Pre-generated test vectors** — random inputs generated before the timed window so `os.urandom()` does not contaminate measurements

### WSL limitations

WSL2 adds ~10-130ns of hypervisor noise depending on system load. This affects:
- **Decapsulation finding** — 37ns delta is borderline, confirmed in 3 of 4 runs
- **Compiler sweep** — results below 50ns delta are marked inconclusive

Production assessment of sub-50ns signals requires bare metal Linux.

---

## Project Structure

```
pqc-scanner/
├── cli.py                      ← all commands
├── requirements.txt
├── scanner/
│   ├── timing.py               ← TVLA engine, warmup, outlier trim, CPU affinity
│   ├── static.py               ← static analysis for crypto context
│   └── report.py               ← JSON and HTML report generation
├── targets/
│   ├── ml_kem_target.py        ← ML-KEM-768 Python harness (kyber-py)
│   ├── ml_dsa_target.py        ← ML-DSA-65 Python harness (dilithium-py)
│   ├── baseline_target.py      ← SHA-256 control experiment
│   └── liboqs_target.py        ← liboqs subprocess interface
└── harness/
    ├── ml_kem_harness.c        ← C timing harness, CLOCK_MONOTONIC_RAW
    └── baseline_harness.c      ← AES-128 noise floor harness
```

---

## Comparison With Existing Tools

| Capability | dudect | Riscure | This tool |
|---|---|---|---|
| ML-KEM / ML-DSA targets | ✗ | ✗ | ✓ |
| liboqs integration | ✗ | ✗ | ✓ |
| Python + C dual layer | ✗ | ✗ | ✓ |
| Automated HTML reporting | ✗ | ✓ (hardware) | ✓ |
| CI/CD exit codes | ✗ | ✗ | ✓ |
| Noise floor validation | ✗ | ✓ (hardware) | ✓ |
| No hardware required | ✓ | ✗ | ✓ |
| Single command scan | ✗ | ✗ | ✓ |
| Open source | ✓ | ✗ | ✓ |

---

## Related Work

- **dudect** — Reparaz et al. (2016). General-purpose constant-time testing library in C. No PQC targets.
- **KyberSlash** — Cryspen et al. (2023). Manual timing analysis finding division-based leak in Kyber reference implementation.
- **TVLA** — Cryptography Research Inc. (2011). Original leakage assessment methodology.
- **ISO 17825** — Testing methods for the mitigation of non-invasive attack classes against cryptographic modules.

---

## Limitations

- WSL2 hypervisor adds noise. Sub-50ns findings require bare metal Linux for confirmation.
- kyber-py and dilithium-py are documented as non-constant-time reference implementations. Python layer findings are expected and not novel.
- liboqs findings are statistically confirmed. Exploitability in a real network attack requires further research.
- Compiler sweep is inconclusive below 50ns on WSL. clang -O3 finding above 50ns is reproducible.

---

## License

MIT License. See `LICENSE` for details.

---

## Collective Qubits

This project was developed as part of [Collective Qubits](https://github.com/Disha231102004) — a quantum computing projects and education organization.
