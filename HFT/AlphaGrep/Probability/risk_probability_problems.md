# AlphaGrep Probability Problems

## Problem 1: Value at Risk (VaR) Calculation
**Difficulty: Medium**

You have a portfolio with the following characteristics:
- Initial value: $10 million
- Daily volatility: 2%
- Expected daily return: 0.1%
- Positions are normally distributed

Calculate the 1-day VaR at 95% and 99% confidence levels using:
1. Parametric method
2. Historical simulation (assuming you have 252 days of historical data)
3. Monte Carlo simulation

**Follow-up**: How would you adjust for fat tails and skewness?

## Problem 2: Correlation Breakdown
**Difficulty: Hard**

During market stress, correlations between assets tend to increase toward 1. You observe:
- Normal market correlation between Stock A and Stock B: 0.3
- Stress market correlation: 0.8
- Probability of stress market: 5%

If you're long $1M in Stock A and short $500K in Stock B:
1. What's your expected P&L in normal markets?
2. What's your expected P&L during stress?
3. What's the probability of losing more than $100K in a single day?

## Problem 3: Options Hedging Strategy
**Difficulty: Hard**

You're market making in options and have the following position:
- Long 1000 call options (strike $100, 30 days to expiry)
- Current stock price: $95
- Implied volatility: 25%
- Risk-free rate: 3%

1. Calculate your Delta, Gamma, Theta, and Vega exposure
2. How many shares should you hedge with to be Delta-neutral?
3. If volatility drops to 20%, what's your P&L?
4. Design a dynamic hedging strategy for the next week

## Problem 4: Bayesian Updating
**Difficulty: Medium**

You have a trading model with the following characteristics:
- Prior probability of market going up: 60%
- Model accuracy when market goes up: 80%
- Model accuracy when market goes down: 70%

Your model predicts the market will go up today. What's the posterior probability that the market actually goes up?

If you observe the market going up 3 days in a row, how does this change your prior for tomorrow?

## Problem 5: Fat-Tail Risk
**Difficulty: Hard**

Historical data shows that extreme market moves (>3 standard deviations) occur:
- Theoretical normal distribution: 0.27% of the time
- Actual market data: 2.1% of the time

You're using a VaR model based on normal distribution. How would you adjust your risk calculations to account for this fat-tail behavior?

**Bonus**: Implement a t-distribution or mixture model approach to better capture tail risk.

## Solutions

### Problem 1: VaR Calculation

**Parametric VaR (95% confidence)**:
- Z-score for 95% confidence = 1.645
- VaR = Portfolio Value × (μ - σ × Z)
- VaR = $10M × (0.001 - 0.02 × 1.645) = $10M × (-0.03189) = $318,900

**Parametric VaR (99% confidence)**:
- Z-score for 99% confidence = 2.326
- VaR = $10M × (0.001 - 0.02 × 2.326) = $10M × (-0.04552) = $455,200

### Problem 4: Bayesian Updating

Using Bayes' theorem:
P(Up|Signal) = P(Signal|Up) × P(Up) / P(Signal)

Where:
- P(Signal|Up) = 0.8
- P(Up) = 0.6
- P(Signal) = P(Signal|Up)×P(Up) + P(Signal|Down)×P(Down) = 0.8×0.6 + 0.3×0.4 = 0.48 + 0.12 = 0.6

Therefore: P(Up|Signal) = (0.8 × 0.6) / 0.6 = 0.8 or 80%
