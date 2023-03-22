# A Better Low-Pass Filter

The two most popular low-pass filters (in DSP lingo) or moving averages
(in the statistics world) are arguably the following:

- Simple moving average (SMA) aka FIR filter
- Exponential moving average (EMA or EWMA) aka first-order lowpass filter

Let's say we want to average over a window length of 5 samples with the SMA
and compare that to a first-order lowpass filter ("LP-1") with a
[time constant](https://en.wikipedia.org/wiki/Time_constant) of also 5:


Both of these filters are extremes at opposite ends of a spectrum:
- The SMA assigns a constant weight to the past 5 samples, but assigns zero
  weight to all earlier samples.
- The LP-1 weights past samples with an exponential decay, so "recent"
  samples get more weight than "old" samples.

Wouldn't it be nice to have a low-pass filter that is somewhere between
those extremes, i.e., something that has the following impulse response?



$$H(s) = \frac{1}{\tau s + 1}$$
