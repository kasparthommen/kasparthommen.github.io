from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.python.zero_pole import ZeroPole


class Lowpass1:
    def __init__(self, tau: float) -> None:
        self.tau = tau
        self.y = 0

    def filter(self, u: float) -> float:
        self.y = (u + self.tau*self.y) / (self.tau + 1)
        return self.y


class NiceLowpass:
    def __init__(self, order: int, tau: float) -> None:
        self.lp1s = [Lowpass1(tau / order) for _ in range(order)]

    def filter(self, u: float) -> float:
        y_sum = 0
        for lp1 in self.lp1s:
            yi = lp1.filter(u)
            y_sum += yi
            u = yi

        return y_sum / len(self.lp1s)


# impulse responses
T = 60
tau = 20
t = np.arange(T+1)


def nice_lowpass(order: int, tau: float) -> ZeroPole:
    nlp = ZeroPole.ZERO
    for i in range(order):
        nlp += ZeroPole.lp1(w_pole=order/tau)**(i+1)

    nlp /= order
    return nlp


lp1 = ZeroPole.lp1(w_pole=1/tau)
nlp4 = nice_lowpass(order=4, tau=tau)

y_sma = np.zeros_like(lp1.impulse(T+1))
y_sma[(t >= 0) & (t < tau)] = 1/tau


# plot & save
def finalize_plot(tau: float, file_name: str) -> None:
    ax.set_xticks(np.linspace(0, T, 7))
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05/tau)
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_zorder(99)
    ax.legend()
    fig.tight_layout()
    plt.savefig(Path('posts/better-lowpass')/file_name)
    plt.show()


drawstyle = 'default'
marker = '.'
figsize = [8, 6]

fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('Simple and exponential moving averages')
ax.plot(t, y_sma, marker=marker, drawstyle=drawstyle, label=f'SMA')
ax.plot(t, lp1.impulse(T+1), marker=marker, drawstyle=drawstyle, label=f'EMA')
finalize_plot(tau, "sma+ema.png")

fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('Adding a "nice low-pass" to the mix')
ax.plot(t, y_sma, marker=marker, drawstyle=drawstyle, label=f'SMA')
ax.plot(t, lp1.impulse(T+1), marker=marker, drawstyle=drawstyle, label=f'EMA')
ax.plot(t, nlp4.impulse(T+1), marker=marker, drawstyle=drawstyle, label=f'NLP-4')
finalize_plot(tau, "sma+ema+nlp4.png")

fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('Nice low-passes with varying order')
ax.plot(t, y_sma, marker=marker, drawstyle=drawstyle, label=f'SMA')
for order in [1, 2, 4, 8, 16]:
    nlp = nice_lowpass(order=order, tau=tau)
    label = f'NLP-{order}'
    if order == 1:
        label += f' = EMA'
    ax.plot(t, nlp.impulse(T+1), marker=marker, drawstyle=drawstyle, label=label)
finalize_plot(tau, "sma+nlps.png")

fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('A "bump" function')
lp4 = ZeroPole.lp1(w_pole=4/tau)**4
label = f'Bump'
ax.plot(t, lp4.impulse(T + 1), marker=marker, drawstyle=drawstyle, label=label)
finalize_plot(tau, "bump.png")
