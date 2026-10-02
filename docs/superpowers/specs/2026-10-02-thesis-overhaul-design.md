# Design Document: Thesis Code and Document Overhaul (SCSLP & FHO)

- **Date**: 2026-10-02
- **Author**: Saddam Aditya Hartanto (Assisted by AI)
- **Topic**: Two-Stage Stochastic Optimization for Electric Bike-Sharing System (SCSLP) in Yogyakarta using Fire Hawk Optimizer (FHO)
- **Institution**: Universitas Gadjah Mada (UGM), Faculty of Mathematics and Natural Sciences (FMIPA)

---

## 1. Executive Summary & Context

The thesis titled **"OPTIMALISASI SISTEM ELECTRIC BIKE-SHARING DI YOGYAKARTA DENGAN PENDEKATAN OPTIMISASI STOKASTIK DUA TAHAP MENGGUNAKAN FIRE HAWK OPTIMIZER"** was successfully defended in 2024. 

The original workspace remained in an unorganized state typical of academic deadlines:
1. Research code consisted of raw Jupyter/Colab/Kaggle script dumps (`fork_of_skripsi_v2.py`, `skripsi_generating_data.py`, `copy_of_bismillah.py`, `another_copy_of_data_jalan.py`) with hardcoded Kaggle paths (`/kaggle/input/...`), broken Colab magic commands (`!pip install`), and missing datasets.
2. The LaTeX workspace was contained inside `SCSLP__Copy_/`, cluttered with over 40 build artifacts (`.aux`, `.log`, `.synctex.gz`, `.fls`), stray legacy code (`Main.java`), and a redundant stub file (`Bab6.tex`).
3. No git repository, no package management (`pyproject.toml` or `uv`), and no reproducibility instructions existed.

The goal of this project is to perform a complete overhaul to elevate the thesis codebase and document to a top-tier open-source portfolio and publication-grade standard.

---

## 2. Goals & Non-Goals

### Goals
- **Full Reproducibility**: Any user can clone the repository, install dependencies via `uv`, run the data generator, and execute the Fire Hawk Optimizer simulation with a single command.
- **Modular Python Architecture**: Clean package `scslp` with strict separation of concerns (configuration, data generator, stochastic formulation & constraints, FHO metaheuristic, Gurobi MIP benchmark, visualization, and CLI).
- **Canonical Datasets**: Generate and commit deterministic, seedable baseline datasets under `data/` (`stations_data.csv`, `scenarios_data.csv`, `distance_matrix.csv`, `config.json`).
- **Pristine LaTeX Repository**: Reorganize `thesis/` with modular chapters (`chapters/`), categorized figures (`figures/`), official signed PDFs (`attachments/`), and automated out-of-source builds (`make` / `latexmk` targeting `build/`).
- **LaTeX Text & Editorial Polish**: Eliminate redundant stubs (`Bab6.tex`), fix appendix listings to point to clean Python code, harmonize mathematical notation and Indonesian academic terminology (KBBI & UGM standards).
- **Portfolio Showcase**: High-impact bilingual `README.md` with mathematical formulations, system architecture diagrams, interactive Folium map screenshots, and convergence plots.

### Non-Goals
- Altering the mathematical model or changing the historical findings/conclusions presented to the thesis examination board in 2024.
- Requiring a paid Gurobi license for standard execution (FHO remains the primary solver; Gurobi solver remains an optional benchmark module).

---

## 3. Architecture & Repository Layout

### Target Directory Layout
```text
skripsi/
├── .gitignore                      # Comprehensive ignores for LaTeX, Python, OS
├── README.md                       # High-impact presentation, math, CLI guide
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-10-02-thesis-overhaul-design.md
├── data/                           # Canonical, seedable datasets
│   ├── stations_data.csv           # 15 candidate stations around Yogyakarta
│   ├── scenarios_data.csv          # Demand scenarios (Weekdays, Saturday, Sunday)
│   ├── distance_matrix.csv         # Inter-station geodesic distance matrix (km)
│   └── config.json                 # Typed problem parameters snapshot
├── code/                           # Standalone Python package
│   ├── pyproject.toml              # Modern uv / PEP 621 packaging
│   ├── README.md                   # Codebase quickstart & API docs
│   ├── src/
│   │   └── scslp/
│   │       ├── __init__.py         # Package exports
│   │       ├── config.py           # Dataclasses: ProblemConfig, FHOConfig
│   │       ├── generator.py        # Seedable station/trip/scenario generator
│   │       ├── problem.py          # Two-stage objective & constraint validator
│   │       ├── fho.py              # Pure NumPy Fire Hawk Optimizer implementation
│   │       ├── gurobi_solver.py    # Exact Two-Stage MIP solver (optional)
│   │       ├── viz.py              # Folium maps and Matplotlib convergence curves
│   │       └── cli.py              # Typer/argparse CLI entry point (`scslp-cli`)
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_generator.py       # Data integrity, coordinate checks
│   │   ├── test_problem.py         # Constraint validation unit tests
│   │   └── test_fho.py             # Optimization loop & convergence tests
│   └── scripts/
│       ├── run_simulation.py       # Full thesis simulation script
│       └── export_thesis_tables.py # Generates LaTeX tabular output
└── thesis/                         # Clean LaTeX workspace
    ├── main.tex                    # Clean root document (refactored Skripsi.tex)
    ├── skripsimathugm.cls          # Official UGM Math class
    ├── setspace.sty                # Spacing package
    ├── .latexmkrc                  # latexmk configuration directing output to build/
    ├── Makefile                    # make pdf, make clean, make view
    ├── chapters/
    │   ├── bab1_pendahuluan.tex    # Background, problem statement, objectives
    │   ├── bab2_landasan_teori.tex # Linear programming, metaheuristics, FHO
    │   ├── bab3_optimisasi_stokastik.tex # Two-stage stochastic programming theory
    │   ├── bab4_pemodelan_simulasi.tex   # SCSLP model formulation & simulation
    │   └── bab5_penutup.tex        # Conclusions and future work
    ├── figures/
    │   ├── logougm.png
    │   ├── fho/                    # FHO flowcharts and mechanism diagrams
    │   └── simulation/             # Map overlays, LINGO comparison, convergence plots
    ├── attachments/
    │   ├── pengesahanskripsi.pdf   # Official examination approval sheet
    │   └── pernyataan.pdf          # Official statement of originality
    └── appendices/
        └── listings/               # Formatted code listings referenced by main.tex
```

---

## 4. Component Design: Python Codebase (`code/`)

### 4.1 Configuration Management (`config.py`)
Encapsulate all simulation constants into strongly typed Python dataclasses:
- `ProblemConfig`:
  - `num_stations`: 15
  - `budget_fraction`: 0.8
  - `bike_cost`: IDR 4,000,000
  - `charging_time`: Discretized charging periods
  - `max_trip_duration`: Maximum trip duration
  - `max_battery_distance`: Maximum range on single charge (km)
  - `scenario_probabilities`: `[0.2, 0.3, 0.5]` for Weekday, Saturday, Sunday
- `FHOConfig`:
  - `pop_size`: Default 50
  - `max_generations`: Default 100
  - `num_fire_hawks`: Ratio or fixed count of territory leaders
  - `random_seed`: Seed for deterministic reproducibility

### 4.2 Data Generator (`generator.py`)
Refactored from `skripsi_generating_data.py`:
- Strategic anchor points around Yogyakarta: Tugu Yogyakarta, Stasiun Tugu, Malioboro, UGM, Monjali, Seturan, Bandara Adisutjipto.
- Computes pairwise geodesic distances using `geopy` or vector haversine.
- Generates realistic trip requests with origin/destination pairs, time windows ($s_k, e_k$), revenue, and battery consumption.
- Capable of saving directly to CSVs/JSON or returning in-memory DataFrames.

### 4.3 Problem Formulation & Feasibility (`problem.py`)
Formalizes the mathematical objective and constraints from Chapter 4:
- **Solution Vector**:
  $$\mathbf{x} = [y_1, \dots, y_M \mid b_1, \dots, b_K \mid z_{1,1}, \dots, z_{S, N_S}]$$
  where:
  - $y_i \in \{0, 1\}$: whether station $i$ is opened.
  - $b_k \in \{0, 1\}$: initial bike allocation.
  - $z_{s, k} \in \{0, 1\}$: whether trip $k$ under scenario $s$ is accepted.
- **Objective Function**:
  $$\max \mathbb{E}_{\xi}\left[ \text{Revenue}(\mathbf{z}) \right] - \sum_{i=1}^M f_i y_i - c_{\text{bike}} \sum_{k=1}^K b_k$$
- **Constraint Validator**:
  - Total investment cost $\le \text{budget}$.
  - Trips only originate and terminate at opened stations ($y_i = 1$).
  - Bike balance conservation across discrete time steps $t \in [0, T_{\max}]$.
  - Battery capacity and recharge duration constraints.

### 4.4 Fire Hawk Optimizer (`fho.py`)
Vectorized implementation of the Fire Hawk Optimizer (Azizi et al., 2023):
1. **Initialization**: Generate initial population within bounding constraints; repair solutions with zero open stations.
2. **Evaluation & Sorting**: Evaluate fitness via `TwoStageProblem.evaluate()`.
3. **Territory Assignment**: Designate top $N_{FH}$ candidates as Fire Hawks; partition remaining candidates (preys) to Fire Hawks based on Euclidean distance.
4. **Position Updates**:
   - Fire Hawk movement towards global best or random exploration.
   - Prey movement towards their assigned Fire Hawk or towards a safe location ($SP$).
5. **Boundary & Feasibility Enforcement**: Discrete thresholding / repair heuristic for binary decision variables.
6. **Convergence History**: Record iteration-by-iteration best fitness and population statistics.

### 4.5 Visualization Suite (`viz.py`)
- `generate_folium_map()`: Interactive HTML map displaying candidate stations (blue/green icons for unbuilt/built) and accepted trip polylines with interactive popups.
- `plot_convergence()`: Clean Matplotlib curve showing iteration vs. expected profit (IDR), formatted for publication and thesis inclusion.
- `plot_station_utilization()`: Bar charts showing bike deployments per station.

### 4.6 Command Line Interface (`cli.py`)
Exposed as `scslp-cli` via `pyproject.toml` console scripts:
- `scslp-cli generate --outdir ./data --seed 42`
- `scslp-cli solve --data ./data --generations 100 --pop-size 50 --out ./results`
- `scslp-cli visualize --results ./results --map-out ./results/map.html`

---

## 5. Component Design: Thesis Organization (`thesis/`)

### 5.1 Directory Restructuring
- Move `Skripsi.tex` $\to$ `thesis/main.tex` (maintain backward-compatible symlink or note).
- Move chapters into `thesis/chapters/`:
  - `Bab1.tex` $\to$ `chapters/bab1_pendahuluan.tex`
  - `Bab2.tex` $\to$ `chapters/bab2_landasan_teori.tex`
  - `Bab3.tex` $\to$ `chapters/bab3_optimisasi_stokastik.tex`
  - `Bab4.tex` $\to$ `chapters/bab4_pemodelan_simulasi.tex`
  - `Bab5.tex` $\to$ `chapters/bab5_penutup.tex`
- Delete unfinished stub `Bab6.tex`.
- Move figures into `thesis/figures/`.
- Move official signed scans into `thesis/attachments/`.

### 5.2 Build Automation (`.latexmkrc` & `Makefile`)
- Configure `.latexmkrc`:
  ```perl
  $out_dir = 'build';
  $pdf_mode = 1;
  $bibtex_use = 2;
  ```
- Configure `Makefile`:
  - `make pdf`: Compiles `main.tex` into `build/main.pdf`.
  - `make clean`: Removes all auxiliary files in `build/`.
  - `make view`: Opens compiled PDF in default viewer.

### 5.3 Code Listing References
Update Appendix listings in `main.tex` to include the clean, modular Python modules:
- Appendix A: Data Generator (`code/src/scslp/generator.py`)
- Appendix B: Two-Stage Model & Constraints (`code/src/scslp/problem.py`)
- Appendix C: Fire Hawk Optimizer Implementation (`code/src/scslp/fho.py`)

---

## 6. Verification & Quality Gates

1. **Python Quality**:
   - `uv run pytest code/tests/` passes with 100% green tests.
   - `uv run scslp-cli generate` creates valid CSVs with correct headers and non-empty rows.
   - `uv run scslp-cli solve` completes and logs non-negative expected profit.
2. **LaTeX Verification**:
   - `cd thesis && make pdf` compiles without fatal errors, generating a clean PDF with valid cross-references and table of contents.
   - Verification of page count and layout consistency with UGM mathematical guidelines.
3. **Repository Cleanliness**:
   - Git status is completely clean; zero untracked `.aux`, `.log`, or `.DS_Store` files.
   - All legacy Kaggle/Colab references removed.

---

## 7. Migration Sequence

1. Initialize git and commit the raw legacy workspace (Done: commit `666ab96`).
2. Save and commit this design document.
3. Invoke `writing-plans` skill to generate the detailed step-by-step implementation plan.
4. Execute implementation plan with review checkpoints.
5. Final verification and repository presentation.
