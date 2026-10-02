from __future__ import annotations

import argparse
import json
from pathlib import Path

from scslp.config import FHOConfig, ProblemConfig
from scslp.fho import FireHawkOptimizer
from scslp.generator import generate_synthetic_data, load_datasets, save_datasets
from scslp.problem import TwoStageProblem
from scslp.viz import create_folium_map, plot_convergence


def cli_generate(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data for Yogyakarta SCSLP.")
    parser.add_argument("--outdir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parsed = parser.parse_args(args)

    cfg = ProblemConfig()
    stations_df, scenarios_df, dist_matrix = generate_synthetic_data(cfg, seed=parsed.seed)
    save_datasets(stations_df, scenarios_df, dist_matrix, cfg, parsed.outdir)
    print(f"Datasets generated successfully in {parsed.outdir}")


def cli_solve(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Solve SCSLP with Fire Hawk Optimizer.")
    parser.add_argument("--datadir", type=str, default="data", help="Directory with datasets")
    parser.add_argument("--outdir", type=str, default="results", help="Directory for output")
    parser.add_argument("--generations", type=int, default=50, help="Max generations")
    parser.add_argument("--pop-size", type=int, default=30, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parsed = parser.parse_args(args)

    st_df, sc_df, dm_df, cfg = load_datasets(parsed.datadir)
    problem = TwoStageProblem(st_df, sc_df, dm_df, cfg)
    fho_cfg = FHOConfig(pop_size=parsed.pop_size, max_generations=parsed.generations, random_seed=parsed.seed)

    print(f"Solving SCSLP (Dim: {problem.dimension}, Pop: {fho_cfg.pop_size}, Gen: {fho_cfg.max_generations})...")
    optimizer = FireHawkOptimizer(problem, fho_cfg)
    result = optimizer.optimize()

    out = Path(parsed.outdir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "best_fitness_idr": result.best_fitness,
        "stations_opened": result.decoded["stations_open"],
        "bikes_allocated": result.decoded["bikes_allocated"],
        "trips_accepted": result.decoded["trips_accepted"],
        "fitness_history": result.fitness_history,
    }
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_convergence(result.fitness_history, out / "convergence.png")
    map_viz = create_folium_map(st_df, sc_df, result.decoded)
    map_viz.save(str(out / "solution_map.html"))

    print(f"Simulation completed. Best fitness: IDR {result.best_fitness:,.2f}")
    print(f"Results saved to {parsed.outdir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCSLP E-Bike Sharing Optimization CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_p = subparsers.add_parser("generate", help="Generate synthetic datasets")
    gen_p.set_defaults(func=cli_generate)

    solve_p = subparsers.add_parser("solve", help="Run FHO optimization")
    solve_p.set_defaults(func=cli_solve)

    parsed, remaining = parser.parse_known_args()
    if parsed.command == "generate":
        cli_generate(remaining)
    elif parsed.command == "solve":
        cli_solve(remaining)


if __name__ == "__main__":
    main()
