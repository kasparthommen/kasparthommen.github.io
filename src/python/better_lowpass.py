import matplotlib.pyplot as plt
import numpy as np


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
t0 = 0
T = 20
tau = 5
n = T - t0
t = np.arange(n) + t0

lp1 = Lowpass1(tau=tau)
nlp2 = NiceLowpass(order=2, tau=tau)
nlp4 = NiceLowpass(order=4, tau=tau)
nlp8 = NiceLowpass(order=8, tau=tau)

y_lp1 = np.zeros_like(t, dtype=float)
y_nlp2 = np.zeros_like(t, dtype=float)
y_nlp4 = np.zeros_like(t, dtype=float)
y_nlp8 = np.zeros_like(t, dtype=float)

for i, ti in enumerate(t):
    u = 1 if ti == 0 else 0

    y_lp1[i] = lp1.filter(u)
    y_nlp2[i] = nlp2.filter(u)
    y_nlp4[i] = nlp4.filter(u)
    y_nlp8[i] = nlp8.filter(u)

y_sma = np.zeros_like(y_lp1)
y_sma[(t >= 0) & (t < tau)] = 1/tau


# plot
for plot_nlp in [False, True]:
    fig, ax = plt.subplots()
    drawstyle = 'default'
    ax.plot(t, y_sma, drawstyle='steps-post', label='SMA')
    ax.plot(t, y_lp1, drawstyle='steps-post', label='LP-1')
    if plot_nlp:
        ax.plot(t, y_nlp4, drawstyle='steps-post', label='A "nice" low-pass')

    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlim(0, 20)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend()
    fig.tight_layout()
    plt.show()
