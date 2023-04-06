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


# A "Nicer" Low-Pass Filter

The two most popular
[low-pass filters](https://en.wikipedia.org/wiki/Low-pass_filter)
(in DSP lingo),
also known as
[moving averages](https://en.wikipedia.org/wiki/Moving_average)
(in statistics terms), are arguably the following:

- Simple moving average aka FIR filter, termed "SMA"
- Exponential (or exponentially-weighted) moving average aka first-order lowpass filter, named "EMA"

Here's a visualization of their respective
[impulse responses](https://en.wikipedia.org/wiki/Impulse_response)
with
an SMA window of 20 and an EMA
[time constant](https://en.wikipedia.org/wiki/Time_constant)
of 20, which corresponds to a
[half-life](https://en.wikipedia.org/wiki/Half-life#Formulas_for_half-life_in_exponential_decay)
of $20 \ln(2) \approx 13.9$:

![SMA & EMA](/posts/nice-lowpass/sma+ema.png)

Both of these filters are extremes:
- The SMA gives equal weight to the past 20 samples and then sharply drops off
  to zero. It would be nice if the transition from non-zero to zero weights were
  smoother
- The EMA weights drop of exponentially (hence the name) and thus lead to a long
  tail. It would be nice if it gave a bit more weight to recent samples and if
  it decayed quicker once it's past the recent samples.
  
Wouldn't it be nice to have a filter that is a middle ground between these two
extremes, something with the green impulse response below?

![SMA, EMA & NLP](/posts/nice-lowpass/sma+ema+nlp4.png)

Before going into details of how to construct such a filter, let's look at some
of its desirable properties:
- It gives more weight to recent samples than the EMA but less than an SMA and
  thus represents a middle ground between the two.
- After the initial phase of giving a lot of weight to recent samples, it decays
  much faster than the EMA (but not to zero like the SMA).

So, how do we construct such a filter? Notice that the green impulse response
looks like it could be constructed as

$$c \left(1 - \int_0^\infty \text{bump}(t) dt\right)$$

where $c$ is a scaling factor and where $\text{bump}(t)$  is something like this:

![SMA, EMA & NLP](/posts/nice-lowpass/bump.png)

It turns out that we can easily construct such a bump function by applying a
simple EMA multiple times! Here's a simple, double and quadruple EMA:

![SMA, EMA & NLP](/posts/nice-lowpass/lps.png)

Did you notice that the quadruple EMA ("LP-4") is identical to the bump
function above?

Let's get a bit more formal now. First, let's assume that we have available the
[discrete transfer function](https://en.wikipedia.org/wiki/Z-transform
of an EMA (or a first-order low-pass filter),

$$H_1 = \text{discretized version of } \frac{1}{\tau s + 1}$$

where we don't care about how this discretization has happened (direct, ZOH,
bilinear). Then, we define repeated applications of these filters as

$$H_n = H_1^n\qquad n \geq 1$$


