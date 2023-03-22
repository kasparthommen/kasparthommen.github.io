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


# The Sharpe Ratio Conundrum
**TL; DR:** The Sharpe ratio computed using the typical definition can be positive while the compounded
return can be negative, and vice-versa. This can be solved by using log returns in the Sharpe ratio
formula instead of simple returns.


## Introduction
The Sharpe ratio is a performance metric in finance.
It is defined as the ratio between the annualized expected (or average) excess return of an investment
divided by the annualized standard deviation of the excess returns. The excess return is defined as
the return of the investment, $r_\text{inv}$, minus the risk-free rate, $r_\text{rf}$.
[Typically](https://en.wikipedia.org/wiki/Sharpe_ratio), the
Sharpe ratio is [expressed](https://www.investopedia.com/terms/s/sharperatio.asp) using the following formula:

$$\textit{SR} = \frac{\operatorname{E}(r_\text{inv} - r_\text{rf})}{\sqrt{\operatorname{Var}(r_\text{inv} - r_\text{rf})}}$$

A high Sharpe ratio means that your investment either has high returns, low volatility,
or both, both of which are desirable properties of an investment. The higher the Sharpe ratio, the smoother the
curve of the investment's value.

For simplicity and without loss of generality, we will now assume that the risk-free rate $r_\text{rf}$ is zero,
which means that we can also drop the subscript from the investment return: $r_\text{inv} \rightarrow r$.


## Implementing the Sharpe ratio formula
For practical implementation, we'll replace the expectation operator $\operatorname{E}(\cdot)$ by the
empirical mean and the denominator by the sample standard deviation. The annual returns of our investment
are denoted $r_i$ depending on prices $p_i$ and $p_{i-1}$ at the end of years $i$ and $i-1$ respectively:

$$
\begin{align}
r_i &\triangleq \frac{p_i-p_{i-1}}{p_{i-1}}\\
\bar r &\triangleq \frac{1}{n} \sum_{i=1}^n r_i \\
\sigma(r) &\triangleq \frac{1}{n} \sqrt{\sum_{i=1}^n (r_i - \bar r)^2}
\end{align}
$$

which leads us to the empirical Sharpe ratio:

$$\textit{SR}_\text{emp} \triangleq \frac{\bar R}{\sigma(R)}$$


## The conundrum
Let us now assume that the year-end prices of our (very volatile,
and ultimately losing) investment had the following year-end prices:

$$p = (100, 150, 90, 135, 81)^\intercal$$

Therefore, the returns are

$$
\begin{align}
r &= ((150-100)/100, (90-150)/150, (135-90)/90, (81-135)/135)^\intercal \\
&= (0.5, -0.4, 0.5, -0.4)^\intercal
\end{align}
$$

Quite a bumpy ride! Let's compute the numerator of the empirical Sharpe ratio (i.e., the mean return):

$$
\begin{align}
\bar r &= \frac{1}{4} (0.5 - 0.4 + 0.5 - 0.4) = 0.05 > 0\\
\Rightarrow \textit{SR}_\text{emp} &> 0
\end{align}
$$

This mean that **the Sharpe ratio is positive, but our investment has lost money**! This doesn't make sense,
at least not to me. The Sharpe ratio should have the *same* sign as the overall return (which is
negative as the prices drops from 100 to 81), but it doesn't.

At this point, you might object, arguing that I should have **compounded the returns** rather than
simply average them. After all, returns are cumulative, and two consecutive 10 % returns lead to
*more* than a 20 % compouned return. So why did I compute the numerator wrong?

Well... did I? The definition of the Sharpe ratio clearly states that the numerator is the
**expected value** of the returns (not the *compounding* of the returns). And expectations
are realized using averages. **Nowhere in the Sharpe ratio formula is it stated that we should
compound returns.**


## Resolution
Does that mean that the Sharpe ratio's definition is wrong? Well, maybe. Or maybe we're just
not using the correct kind of returns. In fact, the definition is consistent (in the above
same-sign sense) if we use **log returns instead of (simple) returns**:

$$
\begin{align}
r_i &\triangleq \operatorname{log}(p_i) - \operatorname{log}(p_{i-1}) \\
\Rightarrow \bar R &= \frac{1}{n} \sum_{i=1}^n (\operatorname{log}(p_i) - \operatorname{log}(p_{i-1})) \\
&= \frac{1}{n} (\operatorname{log}(p_n) - \operatorname{log}(p_0))
\end{align}
$$

Note that the last equation corresponds to the annualized compounded log return. Hence, the Sharpe
ratio computed with log returns has the same sign as the (annualized compounded log) return.
Conundrum resolved!


## Conclusion
Use log returns when computing Sharpe ratios to prevent inconsistent signs between the Sharpe ratio
and the compounded return.
