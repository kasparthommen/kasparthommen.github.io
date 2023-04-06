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

I set out to find a new kind of low-pass filter that combines the best of
the following filter types:

- A simple moving average ("SMA") filter (aka equal-weighted FIR filter),
  which is easy to implement (ring buffer, little computation), has limited
  "tail weights" (weights are zero beyond the filter length), but which
  has an unnatural transition from assigning non-zero weights for all
  samples in the kernel to zero for the next sample and beyond.

- A
  [first-order order low-pass](https://en.wikipedia.org/wiki/Low-pass_filter#First_order)
  ("LP-1") filter, which is
  cheap to implement and store, but which doesn't given enough weight to
  "recent" samples and has a long tail due to its exponential decay.

These two filter types are shown below. The SMA has a kernel length of 20
and the LP-a has a
[time constant](https://en.wikipedia.org/wiki/Time_constant)
of 20, which corresponds to a
[half-life](https://en.wikipedia.org/wiki/Half-life#Formulas_for_half-life_in_exponential_decay)
of $20 \ln(2) \approx 13.9$:

![SMA & EMA](/posts/nice-lowpass/sma+ema.png)


The new filter, termed **Nice Low-Pass**, should have an impulse response
that "lies between" the SMA and LP-1:

![SMA, EMA & NLP](/posts/nice-lowpass/sma+ema+nlp4.png)

Let us now specify the requirements for the nice low-pass in more detail:

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

- The impulse response should start flat (similar to a simple moving
  average) to reflect the fact that recent samples should have high
  and similar weight.

- The impulse response should monotonically decay (to zero), because
  older samples should get less weight than more recent ones.

- The impulse response should have a single inflection point, i.e.,
  it should not "wiggle".

- No over- or undershooting.


How do we construct such a filter? Notice that the green impulse response
looks like it could be constructed as

$$c \left(1 - \int_0^\infty \text{bump}(t) dt\right)$$

where $c$ is a scaling factor and where $\text{bump}(t)$ is something like this:

![SMA, EMA & NLP](/posts/nice-lowpass/bump.png)

We can easily construct such a bump function by repeated filtering
using first-order low-pass filters:

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

$$H_\text{LP-n}(s) = H_\text{LP-1}(s)^n \frac{1}{(\tau s + 1)^n} \quad n \geq 2$$.

Using the integral equation above, we can now formally define the
transfer function of a nice low-pass filter of order $n \geq 2$
based on a "bump" of order $n$ as

$$H_\text{NLP-n}(s) = c \frac{1}{s} \left(1 - H_\text{LP-n} \right)$$



