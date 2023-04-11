from __future__ import annotations

from enum import Enum
from typing import Union, Any, List
import numpy as np
from scipy.signal import dimpulse, impulse

ARRAY = Union[List[float], np.ndarray]
COMPLEX_ARRAY = Union[List[float], List[complex], np.ndarray]
RTOL = 1e-12
ATOL = 1e-12


def _poly(a: COMPLEX_ARRAY) -> np.ndarray:
    if np.all(np.isreal(a)):
        poly = np.poly(a)
    else:
        # combine complex conjugate pairs if present
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


def _matching_entries(
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


class ZeroPoleType(Enum):
    ANALOG = "analog"
    DIGITAL = "digital"


class ZeroPole:
    """
    If analog (no delay allowed, but pure differentiators):

                   ┬┬  ╭                 ╮
                   ││  │1 - zeros[i] * s │
                       ╰                 ╯    diffs
    H(s) = gain * ──────────────────────── * s
                   ┬┬  ╭                 ╮
                   ││  │1 - poles[j] * s │
                       ╰                 ╯


    If digital (delay allowed, but)

                   ┬┬  ╭                -1 ╮
                   ││  │1 - zeros[i] * z   │
                       ╰                   ╯    -delay
    H(z) = gain * ────────────────────────── * z
                   ┬┬  ╭                -1 ╮
                   ││  │1 - poles[j] * z   │
                       ╰                   ╯
    """

    ZERO_ANALOG: ZeroPole
    ZERO_DIGITAL: ZeroPole

    ONE_ANALOG: ZeroPole
    ONE_DIGITAL: ZeroPole

    DIFF_ANALOG: ZeroPole
    DIFF_DIGITAL: ZeroPole

    DELAY_DIGITAL: ZeroPole

    @classmethod
    def analog(
            cls,
            gain: float,
            zeros: ARRAY,
            poles: ARRAY,
            diffs: int = 0
    ) -> ZeroPole:
        return ZeroPole(
            type=ZeroPoleType.ANALOG,
            gain=gain,
            zeros=zeros,
            poles=poles,
            delay_or_diffs=diffs)

    @classmethod
    def digital(
            cls,
            gain: float,
            zeros: ARRAY,
            poles: ARRAY,
            delay: int = 0
    ) -> ZeroPole:
        return ZeroPole(
            type=ZeroPoleType.DIGITAL,
            gain=gain,
            zeros=zeros,
            poles=poles,
            delay_or_diffs=delay)

    def __init__(
            self,
            type: ZeroPoleType,
            gain: float,
            zeros: ARRAY,
            poles: ARRAY,
            delay_or_diffs: int = 0
    ) -> None:
        assert type in ZeroPoleType
        if type == ZeroPoleType.DIGITAL:
            assert delay_or_diffs >= 0  # remain causal

        zeros = np.sort(zeros)
        poles = np.sort(poles)

        # assert not np.any(np.iscomplex(zeros))
        # assert not np.any(np.iscomplex(poles))

        zeros, poles = _matching_entries(zeros, poles, remove=True)

        self.type = type
        self.gain = gain
        self.poles = np.real_if_close(poles, tol=1_000_000)
        self.zeros = np.real_if_close(zeros, tol=1_000_000)
        self.delay_or_diffs = delay_or_diffs

    def __neg__(self) -> ZeroPole:
        return ZeroPole(self.type, -self.gain, self.zeros, self.poles, self.delay_or_diffs)

    def __radd__(self, other: Any) -> ZeroPole:
        return self + other

    def __sub__(self, other: Any) -> ZeroPole:
        return self + -other

    def __rsub__(self, other: Any) -> ZeroPole:
        return -self + other

    def __add__(self, other: Any) -> ZeroPole:
        if isinstance(other, float) or isinstance(other, int):
            return self + ZeroPole(self.type, other, [], [])
        elif isinstance(other, ZeroPole):
            self._ensure_same_type(other)

            # compute new poles
            poles_to_add_left, poles_to_add_right = _matching_entries(
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
            num_left = np.pad(num_left, (self.delay_or_diffs, 0))
            num_right = np.pad(num_right, (other.delay_or_diffs, 0))
            left_minus_right = len(num_left) - len(num_right)
            num_left = np.pad(num_left, (0, max(0, -left_minus_right)))
            num_right = np.pad(num_right, (0, max(0, left_minus_right)))
            poly = num_left + num_right
            poly = np.trim_zeros(poly, trim='b')
            if len(poly) == 0:
                return ZeroPole.ZERO_ANALOG if self.type == ZeroPoleType.ANALOG else ZeroPole.ZERO_DIGITAL

            poly_clean = np.trim_zeros(poly, trim='f')
            delay = len(poly) - len(poly_clean)
            zeros = np.roots(poly_clean)  # may become imaginary, which is ok
            gain = poly_clean[0] / _poly(zeros)[0]
            if gain == 0:
                return ZeroPole.ZERO_ANALOG if self.type == ZeroPoleType.ANALOG else ZeroPole.ZERO_DIGITAL

            return ZeroPole(self.type, gain, zeros, poles, delay)
        else:
            raise Exception()

    def __mul__(self, other: Any) -> ZeroPole:
        if isinstance(other, float) or isinstance(other, int):
            return ZeroPole(self.type, self.gain * other, self.zeros, self.poles, self.delay_or_diffs)
        elif isinstance(other, ZeroPole):
            self._ensure_same_type(other)
            return ZeroPole(
                type=self.type,
                gain=self.gain * other.gain,
                zeros=np.concatenate([self.zeros, other.zeros]),
                poles=np.concatenate([self.poles, other.poles]),
                delay_or_diffs=self.delay_or_diffs + other.delay_or_diffs
            )
        else:
            raise Exception()

    def __rmul__(self, other: Any) -> ZeroPole:
        return self * other

    def __truediv__(self, other: Any) -> ZeroPole:
        if isinstance(other, (float, int)):
            return ZeroPole(
                type=self.type,
                gain=self.gain / other,
                zeros=self.zeros,
                poles=self.poles,
                delay_or_diffs=self.delay_or_diffs
            )
        elif isinstance(other, ZeroPole):
            self._ensure_same_type(other)
            delay = self.delay_or_diffs - other.delay_or_diffs
            if delay < 0:
                raise Exception('Result of division would be non-causal')
            return ZeroPole(
                type=self.type,
                gain=self.gain / other.gain,
                zeros=np.concatenate([self.zeros, other.poles]),
                poles=np.concatenate([self.poles, other.zeros]),
                delay_or_diffs=delay
            )
        else:
            raise TypeError(f"Unknown type: {type(other)}")

    def __rtruediv__(self, other: Any) -> ZeroPole:
        if isinstance(other, float) or isinstance(other, int):
            return ZeroPole(self.type, other / self.gain, self.poles, self.zeros, self.delay_or_diffs)
        # elif isinstance(other, ZeroPole):
        #     self._ensure_same_type(other)
        #     return ZeroPole(
        #         type=self.type,
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
            return ZeroPole.ONE_ANALOG if self.type == ZeroPoleType.ANALOG else ZeroPole.ONE_DIGITAL

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
        if self.type == ZeroPoleType.ANALOG:
            num, den = self.num_den()
            sys = (num[::-1], den[::-1])  # num, den (reverse order)
            t, y = impulse(system=sys, T=np.arange(n))
            return y
        elif self.type == ZeroPoleType.DIGITAL:
            sys = (*self.num_den(), 1)  # num, den, dt
            t, y = dimpulse(system=sys, t=np.arange(n))
            return y[0].squeeze()
        else:
            raise Exception

    def num_den(self) -> tuple[np.ndarray, np.ndarray]:
        num = self.gain * np.pad(_poly(self.zeros), (self.delay_or_diffs, 0))
        den = _poly(self.poles)
        num_minus_den = len(num) - len(den)
        num = np.pad(num, (0, max(0, -num_minus_den)))
        den = np.pad(den, (0, max(0, num_minus_den)))
        return num, den

    def magnitude(self, w: np.ndarray) -> np.ndarray:
        num, den = self.num_den()
        s_or_z = self._compute_s_or_z(w)
        n = np.polyval(num[::-1], s_or_z)
        d = np.polyval(den[::-1], s_or_z)
        response = n / d
        return np.abs(response)

    def phase(self, w: np.ndarray) -> np.ndarray:
        num, den = self.num_den()
        s_or_z = self._compute_s_or_z(w)
        n = np.polyval(num[::-1], s_or_z)
        d = np.polyval(den[::-1], s_or_z)
        response = n / d
        return np.arctan2(response.imag, response.real)

    def _compute_s_or_z(self, w: np.ndarray) -> np.ndarray:
        if self.type == ZeroPoleType.ANALOG:
            # s -> j * w
            return 1j * w
        elif self.type == ZeroPoleType.DIGITAL:
            # z^-1 -> exp(-j * w)
            return np.exp(-1j * w)
        else:
            raise Exception

    @classmethod
    def zero(cls, type: ZeroPoleType) -> ZeroPole:
        return ZeroPole.ZERO_ANALOG if type == ZeroPoleType.ANALOG else ZeroPole.ZERO_DIGITAL

    @classmethod
    def one(cls, type: ZeroPoleType) -> ZeroPole:
        return ZeroPole.ONE_ANALOG if type == ZeroPoleType.ANALOG else ZeroPole.ONE_DIGITAL

    @classmethod
    def lp1(cls, type: ZeroPoleType, tau: float) -> ZeroPole:
        assert type in ZeroPoleType
        if type == ZeroPoleType.ANALOG:
            # H(s) = 1 / (tau*s + 1) = 1 / (1 - (-tau)*s)
            return ZeroPole.analog(
                gain=1,
                zeros=[],
                poles=[-tau],
            )
        elif type == ZeroPoleType.DIGITAL:
            # H(z1) = h / (tau+h - tau*z1)
            return ZeroPole.digital(
                gain=1 / (tau + 1),
                zeros=[],
                poles=[tau / (tau + 1)]
            )
        else:
            raise Exception

    @classmethod
    def hp1(cls, type: ZeroPoleType, tau: float) -> ZeroPole:
        assert type in ZeroPoleType
        if type == ZeroPoleType.ANALOG:
            # H(s) = (tau * s) / (tau * s + 1) = tau * s / (1 - (-tau)*s)
            return ZeroPole.analog(
                gain=tau,
                zeros=[],
                poles=[-tau],
                diffs=1
            )
        elif type == ZeroPoleType.DIGITAL:
            return ZeroPole.digital(
                gain=tau / (1 + tau),
                zeros=[1],
                poles=[tau / (1 + tau)]
            )
        else:
            raise Exception

    def order(self) -> int:
        return max(len(self.zeros), len(self.poles))

    def _ensure_same_type(self, other):
        if self.type != other.type:
            raise TypeError('Cannot add analog and digital transfer functions')

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, float) or isinstance(other, int):
            other = ZeroPole(self.type, other, [], [])

        if isinstance(other, ZeroPole):
            return self.type == other.type \
                and np.isclose(self.gain, other.gain, rtol=RTOL, atol=ATOL) \
                and self._allclose(self.zeros, other.zeros, rtol=RTOL, atol=ATOL) \
                and self._allclose(self.poles, other.poles, rtol=RTOL, atol=ATOL) \
                and self.delay_or_diffs == other.delay_or_diffs

        return False

    def __repr__(self) -> str:
        if self.type == ZeroPoleType.ANALOG:
            return f'{ZeroPole.__name__}-{self.type}(gain={self.gain}, zeros={self.zeros}, poles={self.poles}, diffs={self.delay_or_diffs})'
        elif self.type == ZeroPoleType.DIGITAL:
            return f'{ZeroPole.__name__}-{self.type}(gain={self.gain}, zeros={self.zeros}, poles={self.poles}, delay={self.delay_or_diffs})'
        else:
            raise Exception

    def __str__(self) -> str:
        num1, num2, num3 = self._compute_side(self.type, self.zeros)
        den1, den2, den3 = self._compute_side(self.type, self.poles)

        gain = f"{self.gain} * "
        w = max(len(num1), len(den1))
        num1, num2, num3 = num1.center(w), num2.center(w), num3.center(w)
        den1, den2, den3 = den1.center(w), den2.center(w), den3.center(w)

        ng = w + len(gain)
        num1, num2, num3 = num1.rjust(ng), num2.rjust(ng), num3.rjust(ng)
        den1, den2, den3 = den1.rjust(ng), den2.rjust(ng), den3.rjust(ng)
        mid = f"{gain}{w * '─'}"
        if self.delay_or_diffs != 0:
            mid += f" * z"
            num3 += f"    -{self.delay_or_diffs}"

        return "\n".join([num1, num2, num3, mid, den1, den2, den3])

    @staticmethod
    def _compute_side(type: ZeroPoleType, x: np.ndarray) -> tuple[str, str, str]:
        if len(x) == 0:
            return " ", "1", ""

        if type == ZeroPoleType.ANALOG:
            row2 = [f"│1 - {xi} * s │" for xi in x]
            row1 = "   ".join([f"╭{(len(r) - 3) * ' '} ╮" for r in row2])
            row3 = "   ".join([f"╰{(len(r) - 2) * ' '}╯" for r in row2])
            row2 = " * ".join(row2)
            assert len(row1) == len(row2)
            assert len(row2) == len(row3)
            return row1, row2, row3
        elif type == ZeroPoleType.DIGITAL:
            row2 = [f"│1 - {xi} * z   │" for xi in x]
            row1 = "   ".join([f"╭{(len(r) - 5) * ' '}-1 ╮" for r in row2])
            row3 = "   ".join([f"╰{(len(r) - 2) * ' '}╯" for r in row2])
            row2 = " * ".join(row2)
            assert len(row1) == len(row2)
            assert len(row2) == len(row3)
            return row1, row2, row3
        else:
            raise Exception

    @staticmethod
    def _allclose(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> bool:
        return len(a) == len(b) and np.allclose(a=a, b=b, rtol=rtol, atol=atol);


ZeroPole.ZERO_ANALOG = ZeroPole(ZeroPoleType.ANALOG, 0, [], [])
ZeroPole.ZERO_DIGITAL = ZeroPole(ZeroPoleType.DIGITAL, 0, [], [])

ZeroPole.ONE_ANALOG = ZeroPole(ZeroPoleType.ANALOG, 1, [], [])
ZeroPole.ONE_DIGITAL = ZeroPole(ZeroPoleType.DIGITAL, 1, [], [])

ZeroPole.DIFF_ANALOG = ZeroPole(ZeroPoleType.ANALOG, 1, [1], [])
ZeroPole.DIFF_DIGITAL = ZeroPole(ZeroPoleType.DIGITAL, 1, [1], [])

ZeroPole.DELAY_DIGITAL = ZeroPole(ZeroPoleType.DIGITAL, 1, [], [], 1)


def create_nlp(type: ZeroPoleType, tau: float, order: int) -> ZeroPole:
    nlp = ZeroPole.ZERO_ANALOG if type == ZeroPoleType.ANALOG else ZeroPole.ZERO_DIGITAL
    for i in range(order):
        nlp += ZeroPole.lp1(type, 1 / (tau/order)) ** (i + 1)

    nlp /= order
    return nlp
