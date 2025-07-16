# AlphaGrep Trading Puzzles

## Puzzle 1: The Efficient Market Maker
**Difficulty: Hard**

You're a market maker in a liquid stock. You observe the following order book:

```
Bids:          | Asks:
$99.98 (500)   | $100.02 (300)
$99.97 (800)   | $100.03 (600)
$99.96 (200)   | $100.04 (400)
```

A large institutional order comes in to buy 1000 shares at market price. You can:
1. Fill the order immediately
2. Break it into smaller chunks and fill over time
3. Hedge your position in futures (correlation = 0.95)

**Question**: What's your optimal strategy to maximize profit while minimizing risk?

**Additional constraints**:
- Your inventory limit: ±2000 shares
- Current position: Long 500 shares
- Average daily volume: 100,000 shares
- Volatility: 2% daily

## Puzzle 2: The Arbitrage Triangle
**Difficulty: Medium**

You observe the following exchange rates:
- EUR/USD = 1.2000
- GBP/USD = 1.3500
- EUR/GBP = 0.8900

**Question**: Is there an arbitrage opportunity? If yes, how would you execute it with $1M capital?

**Follow-up**: What if there are transaction costs of 0.02% per trade?

## Puzzle 3: The Volatility Surface
**Difficulty: Hard**

You're trading options and notice an arbitrage in the volatility surface:

| Strike | 30-day IV | 60-day IV |
|--------|-----------|-----------|
| $95    | 20%       | 22%       |
| $100   | 18%       | 25%       |
| $105   | 22%       | 24%       |

Current stock price: $100

**Question**: Which options are mispriced and how would you construct a portfolio to exploit this?

## Puzzle 4: The Correlation Trade
**Difficulty: Hard**

You have access to three assets:
- Asset A: Expected return 8%, Volatility 15%
- Asset B: Expected return 12%, Volatility 25%
- Asset C: Expected return 10%, Volatility 20%

Correlations:
- A-B: 0.3
- A-C: 0.5
- B-C: 0.7

You can only invest in two assets with equal weights.

**Question**: Which pair gives you the best risk-adjusted return?

## Puzzle 5: The Market Timing Challenge
**Difficulty: Medium**

You have a model that predicts market direction with 65% accuracy. Each prediction costs $1000 to generate. If correct, you make $10,000. If wrong, you lose $8,000.

**Question**: 
1. What's your expected profit per prediction?
2. How many predictions do you need to be 95% confident of making money?
3. What if the accuracy drops to 55%?

## Puzzle 6: The Liquidity Puzzle
**Difficulty: Hard**

You need to sell 50,000 shares of a stock with the following characteristics:
- Average daily volume: 200,000 shares
- Current bid-ask spread: $0.05
- Price impact: 0.1% per 1% of daily volume traded
- Time constraint: Must complete within 2 days

**Question**: Design an optimal execution strategy (TWAP, VWAP, or custom algorithm).

---

## Solutions

### Puzzle 1: The Efficient Market Maker

**Optimal Strategy**: Break the order into chunks
1. Fill 300 shares immediately at $100.02 (clearing first ask level)
2. Post remaining 700 shares as a limit order at $100.025
3. Hedge 70% of position in futures immediately
4. Adjust hedge dynamically as position fills

**Reasoning**: 
- Immediate fill captures spread but creates large inventory risk
- Chunking reduces market impact and allows better average price
- Futures hedge reduces risk (correlation 0.95 means 95% of risk hedged)

### Puzzle 2: The Arbitrage Triangle

**Check for arbitrage**:
Cross rate: EUR/GBP = (EUR/USD) / (GBP/USD) = 1.2000 / 1.3500 = 0.8889

Market rate: EUR/GBP = 0.8900

**Arbitrage exists**: Market rate > Cross rate

**Execution with $1M**:
1. Sell EUR/GBP: Sell €1,123,596 for £1,000,000
2. Buy GBP/USD: Sell £1,000,000 for $1,350,000  
3. Buy EUR/USD: Buy €1,125,000 with $1,350,000

**Profit**: €1,125,000 - €1,123,596 = €1,404 ≈ $1,685

### Puzzle 5: Market Timing Challenge

**Expected profit per prediction**:
E[Profit] = 0.65 × $10,000 + 0.35 × (-$8,000) - $1,000
E[Profit] = $6,500 - $2,800 - $1,000 = $2,700

**For 95% confidence**:
Using Central Limit Theorem, need approximately 25-30 predictions to be statistically confident of positive returns.

**At 55% accuracy**:
E[Profit] = 0.55 × $10,000 + 0.45 × (-$8,000) - $1,000 = $1,900
Still profitable but lower margin of safety.
