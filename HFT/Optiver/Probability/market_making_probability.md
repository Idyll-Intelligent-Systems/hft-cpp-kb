# Optiver Probability Problems

## Problem 1: Options Market Making Probability
**Difficulty: Hard**

You're market making in options with the following scenario:
- Stock price: $100
- Option strike: $105
- Days to expiry: 30
- Your bid/ask spread: $0.20
- Daily volume: 1000 contracts
- Your market share: 30%

Given that you have a 60% win rate on your trades (due to favorable selection), but adverse selection occurs 40% of the time with an average loss of $0.50 per contract:

1. What's your expected P&L per contract?
2. What's the probability of losing money on any given day?
3. How does increasing your spread to $0.30 change these metrics?

## Problem 2: Flash Crash Probability
**Difficulty: Hard**

Historical data shows:
- Normal market moves: 95% of days (±2% max move)
- Moderate stress: 4% of days (±5% max move)  
- Flash crash: 1% of days (±10% max move)

You have a market making strategy with:
- Normal day P&L: +$1000 average
- Stress day P&L: -$2000 average
- Flash crash P&L: -$15000 average

1. What's your expected daily P&L?
2. What's the probability of a monthly loss (20 trading days)?
3. How much capital do you need for 99% confidence of surviving 1 year?

## Problem 3: Cointegration Breaking
**Difficulty: Medium**

You have a pairs trading strategy between two ETFs that are usually cointegrated. The relationship breaks down once every 6 months on average, lasting 3 days.

During normal periods:
- Daily P&L: Normal distribution, μ = $500, σ = $200
- Win rate: 65%

During breakdown periods:
- Daily P&L: Normal distribution, μ = -$1500, σ = $800
- Win rate: 25%

1. What's the probability the relationship is broken on any given day?
2. If you observe 3 consecutive losing days, what's the probability you're in a breakdown period?
3. Should you stop trading after 2 consecutive losing days?

## Problem 4: Latency Arbitrage
**Difficulty: Hard**

Your HFT system has:
- Signal generation: 100 microseconds
- Order transmission: 50 microseconds  
- Exchange processing: 200 microseconds (variable)

Market opportunities last on average 500 microseconds. Your success rate depends on how early you arrive:
- First 25%: 90% success rate, $50 profit per trade
- Next 25%: 70% success rate, $30 profit per trade
- Next 25%: 40% success rate, $10 profit per trade
- Last 25%: 10% success rate, $5 profit per trade

If exchange processing time is exponentially distributed with λ = 0.005 (per microsecond):

1. What's your overall expected profit per opportunity?
2. How much would reducing signal generation time to 50 microseconds increase your expected profit?
3. What's the maximum you should pay for a 10-microsecond latency improvement?

## Problem 5: Risk Management Sizing
**Difficulty: Medium**

You're trading with Kelly criterion position sizing. Your strategy has:
- Win rate: 55%
- Average win: $100
- Average loss: $80
- Accuracy of win/loss prediction: 70%

1. What's the optimal Kelly fraction?
2. How does the prediction accuracy affect optimal sizing?
3. If you use half-Kelly, what's your expected growth rate?

---

## Solutions

### Problem 1: Options Market Making

**Expected P&L per contract:**
- Favorable selection (60%): +$0.10 (half spread)
- Adverse selection (40%): -$0.50

Expected P&L = 0.6 × $0.10 + 0.4 × (-$0.50) = $0.06 - $0.20 = -$0.14

This suggests the current strategy is unprofitable!

**With $0.30 spread:**
- Favorable selection: +$0.15
- Expected P&L = 0.6 × $0.15 + 0.4 × (-$0.50) = $0.09 - $0.20 = -$0.11

Still negative, indicating need for better selection or wider spreads.

### Problem 2: Flash Crash Analysis

**Expected daily P&L:**
E[P&L] = 0.95 × $1000 + 0.04 × (-$2000) + 0.01 × (-$15000)
E[P&L] = $950 - $80 - $150 = $720

**Probability of monthly loss:**
Using normal approximation (Central Limit Theorem):
- Monthly expected: $720 × 20 = $14,400
- Monthly std dev: σ√20 ≈ $7,000 (assuming daily σ ≈ $1,566)
- P(Monthly loss) = P(Z < -14,400/7,000) ≈ P(Z < -2.06) ≈ 2%

### Problem 4: Latency Arbitrage

**Expected arrival time:**
Total latency = 100 + 50 + Exponential(λ=0.005) = 150 + Exponential(0.005)

Expected exchange processing = 1/λ = 200 microseconds
Expected total latency = 350 microseconds

**Success probability:**
Since opportunities last 500 microseconds on average, and our expected latency is 350 μs, we arrive in the first 70% of the opportunity window.

This corresponds to roughly 70% success rate with average profit around $35.

**Expected profit per opportunity:**
≈ 0.70 × $35 = $24.50

### Problem 5: Kelly Criterion

**Optimal Kelly fraction:**
f* = (bp - q) / b
where b = 100/80 = 1.25, p = 0.55, q = 0.45

f* = (1.25 × 0.55 - 0.45) / 1.25 = (0.6875 - 0.45) / 1.25 = 0.19

**With prediction accuracy of 70%:**
The effective win rate becomes less reliable, suggesting a reduced Kelly fraction of approximately 0.19 × 0.70 = 0.133 or 13.3%.

**Half-Kelly growth rate:**
Using half the optimal fraction (9.5%) provides more conservative growth with significantly reduced volatility while maintaining most of the return.
