from scslp.config import FHOConfig, ProblemConfig
from scslp.fho import FireHawkOptimizer
from scslp.generator import generate_synthetic_data
from scslp.problem import TwoStageProblem


def test_fho_optimization_runs():
    cfg = ProblemConfig(num_stations=15)
    st_df, sc_df, dm_df = generate_synthetic_data(cfg, seed=42)
    problem = TwoStageProblem(st_df, sc_df, dm_df, cfg)

    fho_cfg = FHOConfig(pop_size=10, max_generations=5, num_fire_hawks=2, random_seed=42)
    optimizer = FireHawkOptimizer(problem, fho_cfg)
    result = optimizer.optimize()

    assert len(result.fitness_history) == 5
    assert result.best_solution.shape == (problem.dimension,)
    assert isinstance(result.best_fitness, float)
    assert result.best_fitness >= 0.0
    assert "stations_open" in result.decoded
    assert "bikes_allocated" in result.decoded
    assert "trips_accepted" in result.decoded
