<script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


# The "Nice" Low-Pass Filter

_March 2023_

## Introduction and Motivation
I have set out to find a new kind of low-pass filter that combines the best of
the following filter types:

- **[Simple moving average ("SMA")](https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average)**
  filter (aka equal-weighted FIR filter):
  - Pros:
    - Easy to implement (ring buffer, little computation)
    - Has limited "tail weights" (weights are zero beyond the filter length)
  - Cons:
    - High storage requirement: $O(n)$ for an $n$-tap filter
    - Has an unnatural transition from assigning non-zero weights for all
      samples in the kernel to zero for the next sample and beyond

- **[First-order order low-pass ("LP-1")](https://en.wikipedia.org/wiki/Low-pass_filter#First_order)**
  filter:
  - Pros:
    - Cheap to implement and store
  - Cons:
    - Doesn't given enough weight to "recent" samples (due to the immediate
      exponential drop-off)
    - Has a long tail (again due to its exponential decay)

These two filter types are shown below. The SMA has a kernel length of 20
and the LP-a has a
[time constant](https://en.wikipedia.org/wiki/Time_constant)
of $\tau=20$, which corresponds to a
[half-life](https://en.wikipedia.org/wiki/Half-life#Formulas_for_half-life_in_exponential_decay)
of $\lambda = 20 \ln(2) \approx 13.9$:

![SMA & EMA](/posts/nice-lowpass/sma+ema.png)


## The Nice Low-Pass filter
The new filter, termed **Nice Low-Pass**, should have an impulse response
that "lies between" the SMA and LP-1:

![SMA, EMA & NLP](/posts/nice-lowpass/sma+ema+nlp4.png)

Let us now specify the requirements for the Nice Low-Pass in more detail:

- It must be cheap in terms of storage and computation, which suggests
  an
  [IIR](https://en.wikipedia.org/wiki/Infinite_impulse_response)
  filter and rules out
  [FIR](https://en.wikipedia.org/wiki/Finite_impulse_response)
  filters because any FIR filter that is not a simple moving average
  (i.e., which has non-constant weights) has computational complexity of
  $O(n)$ if implemented naively, or $O(\log(n))$ when implemented through
  [FFTs](https://en.wikipedia.org/wiki/Fast_Fourier_transform)
  (which is a non-trivial endeavour).

- The impulse response should start flat (similar to a SMA) to reflect
  the fact that recent samples should have high and similar weight.

- The impulse response should monotonically decay (to zero), because
  older samples should get less weight than more recent ones.

- The impulse response should have a single inflection point, i.e.,
  it should not "wiggle".

- No over- or undershooting.


## Filter design
How do we construct such a filter? Notice that the green impulse response
looks like it could be constructed as

$$c \left(1 - \int_0^\infty \text{bump}(t) dt\right)$$

where $c$ is an appropriate scaling factor and where $\text{bump}(t)$ is
something like this:

![SMA, EMA & NLP](/posts/nice-lowpass/bump.png)

We can easily construct such a bump function by repeated filtering
an impulse using first-order low-pass filters connected in series:

![SMA, EMA & NLP](/posts/nice-lowpass/lps.png)

Did you notice that LP-4 is identical to the bump function above?


Let's get a bit more formal. First, let us switch to the time domain,
where the
[Laplace](https://de.wikipedia.org/wiki/Laplace-Transformation)
[transfer function](https://en.wikipedia.org/wiki/Transfer_function)
of a first-order low-pass filter is defined as

$$H_\text{LP-1}(s) = \frac{1}{\tau s + 1}$$

where $\tau$ is the filter's
[time constant](https://en.wikipedia.org/wiki/Time_constant).
Higher-order filters (whose impulse responses are all "bumps" as
we have seen above), can be expressed as

$$H_\text{LP-n}(s) = H_\text{LP-1}(s)^n = \frac{1}{(\tau s + 1)^n} \quad n \geq 2.$$

Using the integral equation above, we can now formally define the
transfer function of a Nice Low-Pass filter of order $n \geq 2$
based on a "bump" of order $n$, $H_\text{LP-n}$, as

$$H_\text{NLP-n}(s) = c \left( \frac{1}{s} - \frac{1}{s} H_\text{LP-n}(s) \right)$$

where the first $\frac{1}{s}$ is the result of the Laplace transform of $1$ and
where the second one results from integrating the bump function. Expanding,
we find

$$H_\text{NLP-n}(s) = c \frac{1}{s} \left(1 - \frac{1}{(\tau s + 1)^n} \right).$$

To find $c$, we impose that the step response of such a filter must settle
at $1$, meaning that at angular frequency $\omega = 0$, which implies
$s = j \omega = 0$, we must have $H_\text{NLP-n}(0) = 1$. Re-arranging, we obtain

$$H_\text{NLP-n}(s) = c \frac{1}{s} \frac{(\tau s + 1)^n - 1}{(\tau s + 1)^n}$$

$$\quad = c \frac{(\tau s + 1)^n - 1}{s(\tau s + 1)^n}$$

from which we can compute the limit for $s\rightarrow 0$ using
[L'Hôpital's rule](https://en.wikipedia.org/wiki/L%27H%C3%B4pital%27s_rule):

$$1 = \lim_{s\rightarrow 0} H_\text{NLP-n}(s)$$

$$\quad = \lim_{s\rightarrow 0} c \frac{n \tau (\tau s + 1)^{n-1}}{(\tau s + 1)^n + s n \tau (\tau s + 1)^{n-1}}$$

$$\quad = c \frac{n\tau}{1 + 0}$$

$$\quad = c n \tau,$$

and thus

$$c = \frac{1}{n \tau}.$$

This leads to

$$H_\text{NLP-n}(s) = \frac{1}{n \tau} \frac{1}{s} \left(1 - \frac{1}{(\tau s + 1)^n} \right).$$

While mathematically correct, such a formulation doesn't lend itself well for
(discretized) implementation as integrating a difference that approaches zero for
low frequencies can quickly lead to unstable result. Fortunately, there is a better
way. After a bit of arithmetic, we find that

$$H_\text{NLP-n}(s) = \frac{1}{n} \frac{1 - \displaystyle \frac{1}{(\tau s + 1)^n}}{\tau s}$$

$$\quad = \frac{1}{n} \frac{\tau s + 1 - \displaystyle \frac{1}{(\tau s + 1)^n} - \tau s}{\tau s}$$

$$\quad = \frac{1}{n} \frac{\tau s + 1 - \displaystyle \frac{1}{(\tau s + 1)^n}}{\tau s + 1 - 1} - 1$$

$$\quad = \frac{1}{n} \frac{1 - \displaystyle \frac{1}{(\tau s + 1)^{n+1}}}{1 - \displaystyle \frac{1}{\tau s + 1}} - 1$$

which, after observing that this is a
[geometric series](https://en.wikipedia.org/wiki/Geometric_series#Sum),
results in

$$H_\text{NLP-n}(s) = \frac{1}{n} \sum_{k=1}^n \frac{1}{(\tau s + 1)^n}$$

or, more explicitly,

$$H_\text{NLP-n}(s) = \frac{1}{n} \sum_{k=1}^n H_\text{LP-n}(s).$$

Thus, we can construct a Nice Low-Pass filter of order $n$ by simply averaging
low-pass filters of orders $1, 2, \dots, n$ - a striking result.


## Implementation
This filter can be implemented in a straightforward manner, namely by passing
input samples through $n$ first-order low-pass filters connected in series
and by adding the outputs of each stage (followed by the division by $n$ at the end).
This means that such a filter requires $n$ states and has computational
complexity of $O(n)$. Unlike a moving average filter, however, which also
requires $n$ states (the array of the past $n$ input samples), the Nice
Low-Pass can achieve much more smoothing for a given number of $n$.


## Frequency and phase response
The chart below shows magnitude and phase responses as
[Bode plots](https://en.wikipedia.org/wiki/Bode_plot).

![Bode plots](/posts/nice-lowpass/bode.png)

The following Bode plot shows the SMA as well for comparison. Note that the
Nice Low-Pass filters lie indeed "between" the LP-1 and the SMA, both in terms
of magnitude and frequency response.

![Bode plots](/posts/nice-lowpass/bode-sma.png)
