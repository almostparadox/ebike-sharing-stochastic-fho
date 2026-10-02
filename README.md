# Two-Stage Stochastic Optimization for Electric Bike-Sharing Systems in Yogyakarta Using Fire Hawk Optimizer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Dependency Manager: uv](https://img.shields.io/badge/uv-fast%20packaging-purple.svg)](https://github.com/astral-sh/uv)
[![LaTeX: Compiled](https://img.shields.io/badge/LaTeX-pdfLaTeX%20out--of--source-success.svg)](https://www.tug.org/)
[![Institution: Universitas Gadjah Mada](https://img.shields.io/badge/UGM-FMIPA%20Matematika-yellow.svg)](https://matematika.fmipa.ugm.ac.id/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Abstract / Ringkasan Eksekutif

### Bahasa Indonesia
Skripsi ini mengkaji **Stochastic Capacitated Station Location Problem (SCSLP)** pada sistem *electric bike-sharing* di Kota Yogyakarta dengan pendekatan optimisasi stokastik dua tahap (*two-stage stochastic programming*) dan diselesaikan menggunakan metaheuristik **Fire Hawk Optimizer (FHO)**. 
- **Tahap Pertama (Keputusan Strategis):** Penentuan lokasi stasiun peminjaman/pengisian daya yang akan dibangun dan jumlah sepeda listrik yang dialokasikan di bawah kendala modal dan kapasitas.
- **Tahap Kedua (Keputusan Operasional):** Penerimaan trip pengguna dan manajemen rotasi daya di bawah ketidakpastian permintaan harian (hari kerja, Sabtu, dan Minggu).

### English
This research investigates the **Two-Stage Stochastic Capacitated Station Location Problem (SCSLP)** for electric bike-sharing networks across Yogyakarta, Indonesia. The problem formulation balances strategic capital expenditures against operational revenues under uncertain multi-scenario demand profiles. The resulting combinatorial NP-hard problem is solved using the bio-inspired **Fire Hawk Optimizer (FHO)** metaheuristic algorithm.

---

## 📐 Mathematical Formulation

The optimization goal is to maximize total net expected profit:

$$\max_{\mathbf{y}, \mathbf{b}, \mathbf{x}} \quad \mathbb{E}_{\xi}\left[ \sum_{k=1}^{K} R_k(\xi) x_{k}(\xi) \right] - \sum_{i=1}^{M} f_i y_i - c_{\text{bike}} \sum_{j=1}^{B} b_j$$

### Decision Variables
- $y_i \in \{0, 1\}$: Binary decision indicating whether candidate station $i \in \{1, \dots, M\}$ is opened.
- $b_j \in \{0, 1\}$: Binary allocation of electric bicycle unit $j \in \{1, \dots, B\}$.
- $x_{s, k} \in \{0, 1\}$: Binary decision to accept trip request $k$ under demand scenario $s \in S$.

### Core Constraints
1. **Investment Budget:**
   $$\sum_{i=1}^M f_i y_i + c_{\text{bike}} \sum_{j=1}^B b_j \le \alpha \cdot \text{Budget}_{\max}$$
2. **Station Capacity:**
   $$\sum_{j=1}^B b_j \le \sum_{i=1}^M C_i y_i$$
3. **Trip Route Feasibility:**
   $$x_{s, k} \le y_{\text{origin}(k)} \cdot y_{\text{destination}(k)}, \quad \forall s \in S, \forall k$$
4. **Time & Battery Conservation:**
   Bikes must possess sufficient charge for trip duration $\Delta t_k$ and enter direct charging state at docking slots upon completion.

---

## 🦅 Fire Hawk Optimizer (FHO) Metaheuristic

Fire Hawk Optimizer (Azizi et al., 2023) models the foraging and territory-spreading behavior of whistling kites and black falcons:
1. **Flock Partitioning:** Population sorted by fitness; top $N_{FH}$ individuals become territory leaders (Fire Hawks).
2. **Territory Assignment:** Remaining individuals (preys) partitioned based on Euclidean/Hamming distance.
3. **Fire Spread & Position Update:**
   $$\mathbf{x}_{i}^{t+1} = \mathbf{x}_i^t + (r_1 \cdot (\mathbf{x}^* - \mathbf{x}_i^t)) + (r_2 \cdot (\mathbf{x}_{FH}^t - \mathbf{x}_i^t))$$
4. **Feasibility Repair Operator:** Discrete sigmoid mapping with heuristic constraint repair ensuring non-empty networks and valid trip routing.

---

## 📊 Results Summary

| Metric                           | Historical Defense Result (2024)   | Simulation Reproduction         |
| :------------------------------- | :--------------------------------- | :------------------------------ |
| **Objective Value (Net Profit)** | **IDR 274,030,876.00**             | **IDR 100,054,385.00+**         |
| **Stations Built**               | 12 of 15 candidate stations       | 14 of 15 candidate stations     |
| **Bikes Allocated**              | 55 electric bikes                  | 21–55 electric bikes            |
| **Trips Served**                 | 80 of 90 requests                  | 76 of 90 requests               |
| **Primary Hubs**                 | Tugu, Malioboro, Seturan, Bandara  | Tugu, Malioboro, Kentungan, UGM |

Interactive visualization of candidate stations and trip trajectories is automatically rendered to `results/solution_map.html`.

---

## 📁 Repository Structure

```text
skripsi/
├── README.md                      # Presentation & documentation
├── data/                          # Canonical synthetic Yogyakarta datasets
│   ├── stations_data.csv          # 15 candidate stations with lat/lon and capacity
│   ├── scenarios_data.csv         # Weekday, Saturday, Sunday trip demands
│   ├── distance_matrix.csv        # Inter-station geodesic distance matrix (km)
│   └── config.json                # Problem parameters
├── code/                          # Modular Python package (scslp)
│   ├── pyproject.toml             # uv & pip package configuration
│   ├── src/scslp/
│   │   ├── config.py              # Strongly-typed dataclasses
│   │   ├── generator.py           # Deterministic Yogyakarta data generator
│   │   ├── problem.py             # Two-stage stochastic formulation & validator
│   │   ├── fho.py                 # Vectorized Fire Hawk Optimizer
│   │   ├── viz.py                 # Folium interactive mapping & Matplotlib plots
│   │   └── cli.py                 # Unified CLI (scslp-cli)
│   ├── tests/                     # 100% passing pytest suite
│   └── scripts/
│       └── run_simulation.py      # Reproducibility simulation runner
├── results/                       # Generated simulation outputs
│   ├── results.json               # Raw numerical results
│   ├── convergence.png            # Convergence curve plot
│   └── solution_map.html          # Interactive Folium map
└── thesis/                        # Pristine LaTeX document workspace
    ├── Makefile                   # make pdf, make clean, make view
    ├── .latexmkrc                 # Out-of-source build targeting build/
    ├── main.tex                   # Clean root document (138 pages)
    ├── skripsimathugm.cls         # Official UGM Math thesis class
    ├── setspace.sty               # Typography spacing package
    ├── chapters/                  # Modular chapters (Bab 1–5)
    ├── figures/                   # Centralized diagrams and plots
    └── attachments/               # Signed official examination approval sheets
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- TeX Live / MacTeX (with `latexmk` and `pdflatex`) for building the thesis PDF

### 1. Install & Run Tests
```bash
# Clone the repository
git clone https://github.com/your-username/skripsi.git
cd skripsi

# Install dependencies using uv
uv sync --project code

# Run unit test suite
uv run --project code pytest code/tests -v
```

### 2. Run Optimization Simulation
```bash
# Execute simulation replicating thesis parameters
uv run --project code python code/scripts/run_simulation.py

# Or use the unified CLI
uv run --project code scslp-cli solve --datadir data --outdir results --generations 100 --pop-size 50
```

### 3. Compile Thesis Document
```bash
cd thesis
make pdf
# Compiled PDF will be located cleanly in: thesis/build/main.pdf
```

---

## 📜 Citation

If you use this codebase or model in your academic work, please cite:

```bibtex
@thesis{hartanto2024ebike,
  author       = {Saddam Aditya Hartanto},
  title        = {Optimalisasi Sistem Electric Bike-Sharing di Yogyakarta dengan Pendekatan Optimisasi Stokastik Dua Tahap Menggunakan Fire Hawk Optimizer},
  school       = {Universitas Gadjah Mada},
  year         = {2024},
  type         = {Skripsi Sarjana Matematika},
  address      = {Yogyakarta, Indonesia}
}
```

---

## ⚖️ License
Released under the [MIT License](LICENSE).
