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

![SMA & EMA](/better-lowpass/sma+ema.png)

Both of these filters are extremes:
- The SMA gives equal weight to the past 20 samples and then sharply drops off
  to zero.
- The EMA weights drop of exponentially (hence the name) and thus lead to a long
  tail.
  
Wouldn't it be nice to have a filter that is a middle ground between these two
extremes, something with the green impulse response below?

![SMA, EMA & NLP](/better-lowpass/sma+ema+nlp4.png)

Before going into details of how to construct such a filter, let's look at some
of its desirable properties:
- It gives more weight to recent samples than the EMA but less than an SMA and
  thus represents a middle ground between the two.
- After the initial phase of giving a lot of weight to recent samples, it decays
  much faster than the EMA (but not to zero like the SMA).

So, how do we construct such a filter? Notice that the green impulse response
looks like it could be constructed as

$$c (1 - \int_0^\infty \text{bump}(t) dt)$$

where $c$ is a scaling factor and where $\text{bump}(t)$  is something like this:

![SMA, EMA & NLP](/better-lowpass/bump.png)





$$H(s) = \frac{1}{\tau s + 1}$$
