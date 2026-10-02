import numpy as np

from scslp.config import ProblemConfig
from scslp.generator import generate_synthetic_data
from scslp.problem import TwoStageProblem


def test_problem_initialization_and_evaluation():
    cfg = ProblemConfig(num_stations=15)
    st_df, sc_df, dm_df = generate_synthetic_data(cfg, seed=42)
    problem = TwoStageProblem(st_df, sc_df, dm_df, cfg)

    assert problem.dimension > 0

    # Trivial solution with no stations open must yield 0 profit
    zero_sol = np.zeros(problem.dimension, dtype=int)
    profit = problem.evaluate(zero_sol)
    assert profit == 0.0
    assert not problem.is_feasible(zero_sol)


def test_constraints_validation():
    cfg = ProblemConfig(num_stations=15)
    st_df, sc_df, dm_df = generate_synthetic_data(cfg, seed=42)
    problem = TwoStageProblem(st_df, sc_df, dm_df, cfg)

    # Open all stations, allocate 1 bike, no trips
    sol = np.zeros(problem.dimension, dtype=int)
    sol[: problem.dim_y] = 1  # all stations open
    sol[problem.dim_y] = 1  # 1 bike
    assert problem.is_feasible(sol)

    # Exceed capacity: allocate more bikes than opened stations capacity
    sol_exceed_cap = np.zeros(problem.dimension, dtype=int)
    sol_exceed_cap[0] = 1  # open station 0 only
    cap_st0 = int(st_df.loc[0, "capacity"])
    # Allocate cap_st0 + 1 bikes
    sol_exceed_cap[problem.dim_y : problem.dim_y + cap_st0 + 1] = 1
    assert not problem.is_feasible(sol_exceed_cap)
