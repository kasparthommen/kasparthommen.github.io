from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.python.zero_pole import ZeroPole, ZeroPoleType
from matplotlib.ticker import MultipleLocator

# impulse responses
type = ZeroPoleType.ANALOG
T = 60
tau = 20
t = np.arange(T+1)


def nice_lowpass(order: int, tau: float) -> ZeroPole:
    nlp = ZeroPole.zero(type)
    for i in range(order):
        nlp += ZeroPole.lp1(type=type, tau=tau/order)**(i+1)

    nlp /= order
    return nlp


lp1 = ZeroPole.lp1(type=type, tau=tau)
nlp4 = nice_lowpass(order=4, tau=tau)

y_sma = np.zeros_like(lp1.impulse(T+1))
y_sma[(t >= 0) & (t < tau)] = 1/tau


# plot & save
def finalize_plot(tau: float, file_name: str, legend: bool = True, y_lim: float = np.nan) -> None:
    ax.set_xticks(np.linspace(0, T, 7))
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05/tau if np.isnan(y_lim) else y_lim)
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_zorder(99)
    if legend:
        ax.legend()
    fig.tight_layout()
    plt.savefig(Path('posts/nice-lowpass')/file_name)
    plt.show()


drawstyle = 'default'
marker = '.'
lw = 0.5
figsize = [8, 6]


# SMA & LP-1
fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('Simple and exponential moving averages')
ax.plot(t, y_sma, marker=marker, drawstyle=drawstyle, lw=lw, label=f'FIR')
ax.plot(t, lp1.impulse(T+1), marker=marker, drawstyle=drawstyle, lw=lw, label=f'LP-1')
finalize_plot(tau=tau, file_name="sma+ema.png")


# # SMA, LP-1 & NLP
fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('Adding a "Nice Low-Pass" to the mix')
ax.plot(t, y_sma, marker=marker, drawstyle=drawstyle, lw=lw, label=f'FIR')
ax.plot(t, lp1.impulse(T+1), marker=marker, drawstyle=drawstyle, lw=lw, label=f'LP-1')
ax.plot(t, nlp4.impulse(T+1), marker=marker, drawstyle=drawstyle, lw=lw, label=f'Nice low-pass')
finalize_plot(tau=tau, file_name="sma+ema+nlp4.png")


# # NLPs
# fig, ax = plt.subplots(figsize=figsize)
# fig.suptitle('Nice low-passes with varying order')
# ax.plot(t, y_sma, marker=marker, drawstyle=drawstyle, lw=lw, label=f'FIR')
# for order in [1, 2, 4, 8, 16]:
#     nlp = nice_lowpass(order=order, tau=tau)
#     label = f'NLP-{order}'
#     if order == 1:
#         label += f' = LP-1'
#     ax.plot(t, nlp.impulse(T+1), marker=marker, drawstyle=drawstyle, lw=lw, label=label)
# finalize_plot(tau=tau, file_name="sma+nlps.png")


# bump
fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('A "bump" function')
lp4 = ZeroPole.lp1(type=type, tau=tau/4)**4
ax.plot(t, lp4.impulse(T + 1), marker=marker, drawstyle=drawstyle, lw=lw)
finalize_plot(tau=tau, file_name="bump.png", legend=False)

# LP-n
fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('LP-n')
for order in [1, 2, 4, 8]:
    lpn = ZeroPole.lp1(type=type, tau=tau/4)**order
    label = f'LP-{order}'
    ax.plot(t, lpn.impulse(T+1), marker=marker, drawstyle=drawstyle, lw=lw, label=label)
finalize_plot(tau=tau, file_name="lps.png", y_lim=4.2/tau)


# Bode plot
for show_sma in [False, True]:
    tau = 100
    w = np.linspace(1e-4/1.1, 1.1, 100000)
    fig, [ax1, ax2] = plt.subplots(figsize=[figsize[0], 1.6*figsize[1]], nrows=2)
    fig.suptitle('Bode plot')
    ax1.set_title('Magnitude response [dB]')
    ax2.set_title('Phase response [deg]')
    ax1.axvline(1e-2, ls='--', c='k', label=r'$\omega_0 = 1/\tau_0$')
    ax2.axvline(1e-2, ls='--', c='k', label=r'$\omega_0 = 1/\tau_0$')
    for order in [1, 2, 4, 8, None]:
        if order is None:
            if show_sma:
                # SMA
                sma = sum([ZeroPole.digital(gain=1, zeros=[], poles=[], delay=i) for i in range(tau)]) / tau
                mag = sma.magnitude(w)
                phase = sma.phase(w)
                label = 'SMA'
            else:
                continue
        else:
            nlp = nice_lowpass(order=order, tau=tau)
            label = f'NLP-{order}' if order >= 2 else 'LP-1'
            mag = nlp.magnitude(w)
            phase = nlp.phase(w)

        mag_db = 10*np.log10(mag)
        ax1.semilogx(w, mag_db, label=label)
        ax2.semilogx(w, phase * 180 / np.pi, label=label)
    ax1.spines[['right', 'top']].set_visible(False)
    ax2.spines[['right', 'top']].set_visible(False)
    ax1.xaxis.set_zorder(99)
    ax2.xaxis.set_zorder(99)
    ax1.legend()
    ax2.legend()
    ax1.yaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(45))
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_ylim([-30, 1])
    fig.tight_layout()
    file_name = 'bode-sma.png' if show_sma else 'bode.png'
    plt.savefig(Path('posts/nice-lowpass') / file_name)
    plt.show()
