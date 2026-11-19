import pytest

from rspect import make_instance, rspect
from rspect.glns import GLNSError


@pytest.fixture(scope="session")
def glns():
    """Skip tests that need Julia + GLNS when they are not installed."""
    try:
        rspect.solve(make_instance(2, 1, seed=0))
    except GLNSError as err:
        pytest.skip(f"Julia/GLNS not available: {err}")


@pytest.fixture(scope="session")
def gurobi():
    """Skip tests that need a working Gurobi installation."""
    try:
        import gurobipy as gp
        gp.Model().dispose()
    except Exception as err:  # missing package or license
        pytest.skip(f"Gurobi not available: {err}")
