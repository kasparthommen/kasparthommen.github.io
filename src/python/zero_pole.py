from __future__ import annotations

from typing import Union, Any, List
import numpy as np
from scipy.signal import dimpulse


ARRAY = Union[List[float], np.ndarray]
COMPLEX_ARRAY = Union[List[float], List[complex], np.ndarray]
RTOL = 1e-12
ATOL = 1e-12


def _poly(a: COMPLEX_ARRAY) -> np.ndarray:
    if np.all(np.isreal(a)):
        poly = np.poly(a)
    else:
        a = np.sort(a)
        poly = [1]
        while len(a) >= 2:
            if np.isclose(a[0], np.conj(a[1])):
                # complex conjugate pair
                re = np.real(a[0])
                im = np.imag(a[0])
                poly = np.polymul(poly, [1, -2 * re, re ** 2 + im ** 2])
                a = a[2:]
            else:
                poly = np.polymul(poly, [1, -np.real_if_close(a[0], tol=1_000_000)])
                a = a[1:]

        if len(a) > 0:
            poly = np.polymul(poly, np.poly(a))

    assert np.all(np.isreal(poly))
    return np.atleast_1d(poly)


def matching_entries(
    a: np.ndarray,
    b: np.ndarray,
    remove: bool = False,
    get: bool = False
) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    assert remove != get

    matches = set(a).intersection(b)
    entries = []
    for m in matches:
        while True:
            ia = np.where(a == m)[0]
            ib = np.where(b == m)[0]
            if len(ia) > 0 and len(ib) > 0:
                a = np.delete(a, ia[0])
                b = np.delete(b, ib[0])
                entries.append(m)
            else:
                break

    return a, b if remove else np.array(entries)


class ZeroPole:
    """
                   ┬┬  ╭                -1 ╮
                   ││  │1 - zeros[i] * z   │
                       ╰                   ╯    -delay
    H(z) = gain * ────────────────────────── * z
                   ┬┬  ╭                -1 ╮
                   ││  │1 - poles[j] * z   │
                       ╰                   ╯
    """

    ZERO: ZeroPole
    ONE: ZeroPole
    DIFF: ZeroPole
    DELAY: ZeroPole

    def __init__(self, gain: float, zeros: ARRAY, poles: ARRAY, delay: int = 0) -> None:
        assert delay >= 0

        zeros = np.sort(zeros)
        poles = np.sort(poles)

        # assert not np.any(np.iscomplex(zeros))
        # assert not np.any(np.iscomplex(poles))

        zeros, poles = matching_entries(zeros, poles, remove=True)

        self.gain = gain
        self.poles = np.real_if_close(poles, tol=1_000_000)
        self.zeros = np.real_if_close(zeros, tol=1_000_000)
        self.delay = delay

    def __neg__(self) -> ZeroPole:
        return ZeroPole(-self.gain, self.zeros, self.poles, self.delay)

    def __radd__(self, other: Any) -> ZeroPole:
        return self + other

    def __sub__(self, other: Any) -> ZeroPole:
        return self + -other

    def __rsub__(self, other: Any) -> ZeroPole:
        return -self + other

    def __add__(self, other: Any) -> ZeroPole:
        if isinstance(other, float) or isinstance(other, int):
            return self + ZeroPole(other, [], [])
        elif isinstance(other, ZeroPole):
            # compute new poles
            poles_to_add_left, poles_to_add_right = matching_entries(
                other.poles,
                self.poles,
                remove=True
            )
            poles = np.concatenate([self.poles, poles_to_add_left])
            poles2 = np.concatenate([other.poles, poles_to_add_right])
            assert set(poles) == set(poles2), f'poles={poles}, poles2={poles2}'

            # compute new zeros
            zeros_left = np.concatenate([self.zeros, poles_to_add_left])
            zeros_right = np.concatenate([other.zeros, poles_to_add_right])
            num_left = self.gain * _poly(zeros_left)
            num_right = other.gain * _poly(zeros_right)
            num_left = np.pad(num_left, (self.delay, 0))
            num_right = np.pad(num_right, (other.delay, 0))
            left_minus_right = len(num_left) - len(num_right)
            num_left = np.pad(num_left, (0, max(0, -left_minus_right)))
            num_right = np.pad(num_right, (0, max(0, left_minus_right)))
            poly = num_left + num_right
            poly = np.trim_zeros(poly, trim='b')
            if len(poly) == 0:
                return ZeroPole.ZERO

            poly_clean = np.trim_zeros(poly, trim='f')
            delay = len(poly) - len(poly_clean)
            zeros = np.roots(poly_clean)  # may become imaginary, which is ok
            gain = poly_clean[0] / _poly(zeros)[0]
            if gain == 0:
                return ZeroPole.ZERO

            return ZeroPole(gain, zeros, poles, delay)
        else:
            raise Exception()

    def __mul__(self, other: Any) -> ZeroPole:
        if isinstance(other, float) or isinstance(other, int):
            return ZeroPole(self.gain * other, self.zeros, self.poles, self.delay)
        elif isinstance(other, ZeroPole):
            return ZeroPole(
                gain=self.gain * other.gain,
                zeros=np.concatenate([self.zeros, other.zeros]),
                poles=np.concatenate([self.poles, other.poles]),
                delay=self.delay + other.delay
            )
        else:
            raise Exception()

    def __rmul__(self, other: Any) -> ZeroPole:
        return self * other

    def __truediv__(self, other: Any) -> ZeroPole:
        if isinstance(other, (float, int)):
            return ZeroPole(self.gain / other, self.zeros, self.poles, self.delay)
        elif isinstance(other, ZeroPole):
            delay = self.delay - other.delay
            if delay < 0:
                raise Exception('Result of division would be non-causal')
            return ZeroPole(
                gain=self.gain / other.gain,
                zeros=np.concatenate([self.zeros, other.poles]),
                poles=np.concatenate([self.poles, other.zeros]),
                delay=delay
            )
        else:
            raise TypeError(f"Unknown type: {type(other)}")

    def __rtruediv__(self, other: Any) -> ZeroPole:
        if isinstance(other, float) or isinstance(other, int):
            return ZeroPole(other / self.gain, self.poles, self.zeros, self.delay)
        # elif isinstance(other, ZeroPole):
        #     return ZeroPole(
        #         gain=self.gain / other.gain,
        #         zeros=np.concatenate([self.zeros, other.poles]),
        #         poles=np.concatenate([self.poles, other.zeros]))
        else:
            raise Exception()

    def __pow__(self, power: Any, modulo=None) -> ZeroPole:
        if not isinstance(power, int):
            raise Exception('power must be an integer')

        if power < 0:
            raise Exception('power must be >= 0')

        if power == 0:
            return ZeroPole.ONE

        zp = self
        result = self
        for _ in range(power - 1):
            result *= zp

        return result

    def static_gain(self) -> float:
        return self.gain * np.sum(_poly(self.zeros)) / np.sum(_poly(self.poles))

    def normalize(self) -> ZeroPole:
        return self / self.static_gain()

    def impulse(self, n: int) -> np.ndarray:
        sys = (*self.num_den(), 1)
        t, y = dimpulse(system=sys, n=n)
        return y[0].squeeze()

    def num_den(self) -> tuple[np.ndarray, np.ndarray]:
        num = self.gain * np.pad(_poly(self.zeros), (self.delay, 0))
        den = _poly(self.poles)
        num_minus_den = len(num) - len(den)
        num = np.pad(num, (0, max(0, -num_minus_den)))
        den = np.pad(den, (0, max(0, num_minus_den)))
        return num, den

    def magnitude(self, w: np.ndarray) -> np.ndarray:
        num, den = self.num_den()
        z = np.exp(w * 1j)
        n = np.polyval(num, z)
        d = np.polyval(den, z)
        return np.abs(n / d)

    @staticmethod
    def lp1(w_pole: float) -> ZeroPole:
        return ZeroPole(w_pole / (1 + w_pole), [], [1 / (1 + w_pole)])

    @staticmethod
    def hp1(w_zero: float) -> ZeroPole:
        return ZeroPole(1 / (1 + w_zero), [1], [1 / (1 + w_zero)])

    def order(self) -> int:
        return max(len(self.zeros), len(self.poles))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, float) or isinstance(other, int):
            other = ZeroPole(other, [], [])

        if isinstance(other, ZeroPole):
            return np.isclose(self.gain, other.gain, rtol=RTOL, atol=ATOL) \
                   and np.allclose(self.zeros, other.zeros, rtol=RTOL, atol=ATOL) \
                   and np.allclose(self.poles, other.poles, rtol=RTOL, atol=ATOL) \
                   and self.delay == other.delay

        return False

    def __repr__(self) -> str:
        return f'{ZeroPole.__name__}(gain={self.gain}, zeros={self.zeros}, poles={self.poles}, delay={self.delay})'

    def __str__(self) -> str:
        num1, num2, num3 = self._compute_side(self.zeros)
        den1, den2, den3 = self._compute_side(self.poles)

        gain = f"{self.gain} * "
        w = max(len(num1), len(den1))
        num1, num2, num3 = num1.center(w), num2.center(w), num3.center(w)
        den1, den2, den3 = den1.center(w), den2.center(w), den3.center(w)

        ng = w + len(gain)
        num1, num2, num3 = num1.rjust(ng), num2.rjust(ng), num3.rjust(ng)
        den1, den2, den3 = den1.rjust(ng), den2.rjust(ng), den3.rjust(ng)
        mid = f"{gain}{w * '─'}"
        if self.delay != 0:
            mid += f" * z"
            num3 += f"    -{self.delay}"

        return "\n".join([num1, num2, num3, mid, den1, den2, den3])

    @staticmethod
    def _compute_side(x: np.ndarray) -> tuple[str, str, str]:
        if len(x) == 0:
            return " ", "1", ""

        row2 = [f"│1 - {xi} * z   │" for xi in x]
        row1 = "   ".join([f"╭{(len(r) - 5) * ' '}-1 ╮" for r in row2])
        row3 = "   ".join([f"╰{(len(r) - 2) * ' '}╯" for r in row2])
        row2 = " * ".join(row2)
        assert len(row1) == len(row2)
        assert len(row2) == len(row3)
        return row1, row2, row3


ZeroPole.ZERO = ZeroPole(0, [], [])
ZeroPole.ONE = ZeroPole(1, [], [])
ZeroPole.DIFF = ZeroPole(1, [1], [])
ZeroPole.DELAY = ZeroPole(1, [], [], 1)


def create_nlp(tau: float, order: int) -> ZeroPole:
    nlp = ZeroPole.ZERO
    for i in range(order):
        nlp += ZeroPole.lp1(1 / (tau/order)) ** (i + 1)

    nlp /= order
    return nlp