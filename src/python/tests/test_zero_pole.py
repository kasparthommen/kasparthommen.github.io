import pytest

import numpy as np

from src.python.zero_pole import ZeroPole, ZeroPoleType, _poly


@pytest.mark.parametrize("type", [ZeroPoleType.ANALOG, ZeroPoleType.DIGITAL])
def test_num_den(type: ZeroPoleType) -> None:
    np.testing.assert_array_equal(ZeroPole(type, 1, [0.3], [0.5]).num_den()[0], [1, -0.3])
    np.testing.assert_array_equal(ZeroPole(type, 1, [0.3], [0.5]).num_den()[1], [1, -0.5])

    np.testing.assert_array_equal(ZeroPole(type, 1, [0.3, 0.4], [0.5]).num_den()[0], [1, -0.3 - 0.4, 0.3 * 0.4])
    np.testing.assert_array_equal(ZeroPole(type, 1, [0.3, 0.4], [0.5]).num_den()[1], [1, -0.5, 0])

    np.testing.assert_array_equal(ZeroPole(type, 1, [0.3], [0.5, 0.6]).num_den()[0], [1, -0.3, 0])
    np.testing.assert_array_equal(ZeroPole(type, 1, [0.3], [0.5, 0.6]).num_den()[1], [1, -0.5 - 0.6, 0.5 * 0.6])

    np.testing.assert_array_equal(ZeroPole(type, 1, [0.3, 0.4], [0.5], delay_or_diffs=1).num_den()[0],
                                  [0, 1, -0.3 - 0.4, 0.3 * 0.4])
    np.testing.assert_array_equal(ZeroPole(type, 1, [0.3, 0.4], [0.5], delay_or_diffs=1).num_den()[1], [1, -0.5, 0, 0])

    np.testing.assert_array_equal(ZeroPole(type, 1, [0.3], [0.5, 0.6], delay_or_diffs=1).num_den()[0], [0, 1, -0.3])
    np.testing.assert_array_equal(ZeroPole(type, 1, [0.3], [0.5, 0.6], delay_or_diffs=1).num_den()[1],
                                  [1, -0.5 - 0.6, 0.5 * 0.6])


# @pytest.mark.filterwarnings("ignore::scipy.signal.filter_design.BadCoefficients")
def test_math() -> None:
    assert ZeroPole.ZERO_ANALOG + 1 == ZeroPole.ONE_ANALOG
    assert ZeroPole.ZERO_DIGITAL + 1 == ZeroPole.ONE_DIGITAL
    assert 1 + ZeroPole.ZERO_ANALOG == ZeroPole.ONE_ANALOG
    assert 1 + ZeroPole.ZERO_DIGITAL == ZeroPole.ONE_DIGITAL
    assert ZeroPole.ONE_ANALOG == ZeroPole.ZERO_ANALOG + 1
    assert ZeroPole.ONE_DIGITAL == ZeroPole.ZERO_DIGITAL + 1

    assert ZeroPole.lp1(ZeroPoleType.ANALOG, 10) == ZeroPole(ZeroPoleType.ANALOG, 1, [], [-10])

    assert ZeroPole.lp1(ZeroPoleType.DIGITAL, 10) == ZeroPole(
        ZeroPoleType.DIGITAL, 1 / (1 + 10), [], [10 / (1 + 10)])

    assert ZeroPole.hp1(ZeroPoleType.ANALOG, 10) == ZeroPole(ZeroPoleType.ANALOG, 10, [], [-10], 1)

    assert ZeroPole.hp1(ZeroPoleType.DIGITAL, 10) == ZeroPole(
        ZeroPoleType.DIGITAL, 10 / (1 + 10), [1], [10 / (1 + 10)])

    for type in ZeroPoleType:
        one = ZeroPole.one(type)

        assert ZeroPole.lp1(type, 10) + ZeroPole.hp1(type, 10) == one
        assert 1 - ZeroPole.lp1(type, 10) == ZeroPole.hp1(type, 10)
        assert 1 - ZeroPole.hp1(type, 10) == ZeroPole.lp1(type, 10)
        assert ZeroPole.lp1(type, 10) / ZeroPole.lp1(type, 10) == one
        assert ZeroPole.hp1(type, 10) / ZeroPole.hp1(type, 10) == one
        assert 1 - (1 - ZeroPole.lp1(type, 10)) == ZeroPole.lp1(type, 10)
        assert 1 - (1 - ZeroPole.hp1(type, 10)) == ZeroPole.hp1(type, 10)

        assert ZeroPole.lp1(type, 10) ** 0 == one
        assert ZeroPole.lp1(type, 10) ** 1 == ZeroPole.lp1(type, 10)
        assert ZeroPole.lp1(type, 10) ** 2 == ZeroPole.lp1(type, 10) * ZeroPole.lp1(type, 10)
        assert ZeroPole.lp1(type, 10) ** 3 == ZeroPole.lp1(type, 10) * ZeroPole.lp1(type, 10) * ZeroPole.lp1(type, 10)

    assert ZeroPole.ONE_DIGITAL * ZeroPole.DELAY_DIGITAL == ZeroPole.DELAY_DIGITAL
    assert ZeroPole.DELAY_DIGITAL ** 2 == ZeroPole(ZeroPoleType.DIGITAL, 1, [], [], 2)

    for type in ZeroPoleType:
        assert 0.2 - (0.2 - ZeroPole.lp1(type, 0.25)) == ZeroPole.lp1(type, 0.25)

    assert ZeroPole(ZeroPoleType.DIGITAL, 1, [], [0.5]) * ZeroPole.DELAY_DIGITAL == ZeroPole(ZeroPoleType.DIGITAL, 1,
                                                                                             [], [0.5], 1)

    np.testing.assert_array_equal(
        ZeroPole(type, 1, [], [0.5]).impulse(10),
        ZeroPole(type, 1, [], [0.5], 1).impulse(11)[1:]
    )
    np.testing.assert_array_equal(
        ZeroPole(type, 1, [], [0.5]).impulse(10),
        ZeroPole(type, 1, [], [0.5], 4).impulse(14)[4:]
    )


def test_lp1_hp1() -> None:
    tau = 100

    num, den = ZeroPole.lp1(type=ZeroPoleType.ANALOG, tau=tau).num_den()
    np.testing.assert_array_equal(num, [1, 0])
    np.testing.assert_array_equal(den, [1, tau])

    num, den = ZeroPole.hp1(type=ZeroPoleType.ANALOG, tau=tau).num_den()
    np.testing.assert_array_equal(num, [0, tau])
    np.testing.assert_array_equal(den, [1, tau])

    num, den = ZeroPole.lp1(type=ZeroPoleType.DIGITAL, tau=tau).num_den()
    np.testing.assert_array_equal(num, [1/(1+tau), 0])
    np.testing.assert_array_equal(den, [1, -tau/(1+tau)])

    num, den = ZeroPole.hp1(type=ZeroPoleType.DIGITAL, tau=tau).num_den()
    np.testing.assert_array_equal(num, [tau/(1+tau), -tau/(1+tau)])
    np.testing.assert_array_equal(den, [1, -tau/(1+tau)])

    for type in ZeroPoleType:
        ir_lp1 = ZeroPole.lp1(type=type, tau=tau).impulse(1000)
        assert len(ir_lp1) == 1000
        assert ir_lp1[0] == pytest.approx(1/tau, rel=0.01)
        assert ir_lp1[1] == pytest.approx(1/tau, rel=0.02)
        assert ir_lp1[tau] == pytest.approx(1/tau * np.exp(-1), rel=0.01)
        assert ir_lp1[2*tau] == pytest.approx(1/tau * np.exp(-2), rel=0.01)
        assert ir_lp1[3*tau] == pytest.approx(1/tau * np.exp(-3), rel=0.01)

        mag = ZeroPole.lp1(type=type, tau=tau).magnitude(w=np.array([0, 1/tau, 8/tau, 64/tau]))
        assert mag[0] == pytest.approx(1)
        assert mag[1] == pytest.approx(1 / np.sqrt(2), rel=1e-2)
        assert mag[2] == pytest.approx(2**-3, rel=0.02)
        assert mag[3] == pytest.approx(2**-6, rel=0.02)


def test_poly() -> None:
    np.testing.assert_array_equal(_poly([]), [1])
    np.testing.assert_array_equal(_poly([1]), [1, -1])
    np.testing.assert_array_equal(_poly([1, 2]), [1, -3, 2])
    np.testing.assert_array_equal(_poly([2 + 1j, 2 - 1j]), np.poly([2 + 1j, 2 - 1j]))
    np.testing.assert_array_equal(_poly([2 + 1j, 2 - 1j]), [1, -4, 5])
    np.testing.assert_array_equal(_poly([3 + 2j, 3 - 2j]), [1, -6, 13])

    np.testing.assert_array_equal(_poly([2 + 1j, 2 - 1j, 3 + 2j, 3 - 2j]), np.poly([2 + 1j, 2 - 1j, 3 + 2j, 3 - 2j]))
    np.testing.assert_array_equal(_poly([2 + 1j, 2 - 1j, 3 + 2j, 3 - 2j]), np.polymul([1, -4, 5], [1, -6, 13]))
