import numpy as np
import pytest

from rspect import Instance, MissionParams, Tour, check_plan, make_instance, tour_robustness
from rspect.problem import flight_time, team_mission_time

P = MissionParams()  # UAV 10 m/s (2 m/s vertical, 100 m altitude -> 50 s take-off), UGV 2.5 m/s


def ground(x, y):
    return np.array([x, y, 0.0])


def test_flight_time_uses_vtol_model():
    path = np.array([[300.0, 400.0, 100.0]])
    # 500 m out and 500 m back at 10 m/s, plus 50 s take-off and 50 s landing
    assert flight_time(ground(0, 0), path, ground(0, 0), P) == pytest.approx(200.0)


def test_mission_time_of_a_single_tour():
    points = np.array([[500.0, 0.0, 100.0]])
    tour = Tour(ground(500, 0), (0,), ground(500, 0))
    # drive 500 m (200 s), fly 100 s, drive 500 m (200 s); no recharge after the last tour
    assert team_mission_time([tour], ground(0, 0), ground(1000, 0), points, P) == pytest.approx(500.0)


def test_recharge_between_tours():
    points = np.array([[0.0, 0.0, 100.0], [10.0, 0.0, 100.0]])
    tours = [Tour(ground(0, 0), (0,), ground(0, 0)), Tour(ground(10, 0), (1,), ground(10, 0))]
    # two 100 s tours; in between, max(4 s drive, 100 s recharge) = 100 s
    assert team_mission_time(tours, ground(0, 0), ground(10, 0), points, P) == pytest.approx(300.0)


def test_trivial_tours_cost_nothing():
    points = np.array([[500.0, 0.0, 100.0]])
    tour = Tour(ground(500, 0), (0,), ground(500, 0))
    unused = Tour(ground(3000, 3000), (), ground(3000, 3000))
    start, end = ground(0, 0), ground(1000, 0)
    assert team_mission_time([unused, tour, unused], start, end, points, P) == pytest.approx(
        team_mission_time([tour], start, end, points, P))
    assert team_mission_time([], start, end, points, P) == pytest.approx(400.0)  # drive only


def test_check_plan_reports_violations():
    far = np.array([[3000.0, 0.0, 100.0]])
    instance = Instance(far, np.zeros((1, 3)), np.zeros((1, 3)))
    round_trip = Tour(ground(0, 0), (0,), ground(0, 0))  # 6 km at 10 m/s + 100 s > 600 s
    assert any("(3c)" in p for p in check_plan([[round_trip]], instance, P))
    assert any("(3b)" in p for p in check_plan([[]], instance, P))
    below = Tour(ground(3000, 0), (0,), ground(3000, 0))
    assert check_plan([[below]], instance, P) == []


def test_tour_robustness():
    points = np.array([[500.0, 0.0, 100.0]])
    tour = Tour(ground(500, 0), (0,), ground(1000, 0))  # flight 150 s, UGV drive 200 s
    assert tour_robustness(tour, points, P) == pytest.approx((450.0, 400.0))


def test_make_instance():
    instance = make_instance(20, 3, seed=1)
    assert instance.points.shape == (20, 3) and np.all(instance.points[:, 2] == 100.0)
    assert np.all(instance.starts == 0.0) and np.allclose(instance.ends, [4000.0, 4000.0, 0.0])
    assert np.array_equal(instance.points, make_instance(20, 3, seed=1).points)
    shuffled = make_instance(20, 3, seed=1, random_endpoints=True)
    assert np.array_equal(shuffled.points, instance.points)  # endpoints use their own random stream
    assert np.all(shuffled.starts[:, 2] == 0.0)
