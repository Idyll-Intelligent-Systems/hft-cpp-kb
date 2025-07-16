# Plutus HFT Puzzles and Brain Teasers

## Classic Trading and Mathematics Puzzles

### Puzzle 1: The Arbitrage Opportunity
**Problem:** You notice the following quotes in the market:
- EUR/USD: 1.1000 - 1.1002
- USD/JPY: 110.00 - 110.05  
- EUR/JPY: 121.00 - 121.10

Is there an arbitrage opportunity? If yes, what's the maximum profit per €1,000,000 traded?

**Solution:**
Check triangular arbitrage: EUR/USD × USD/JPY should equal EUR/JPY

Cross rate calculation:
- Buy EUR/USD at 1.1002, sell USD/JPY at 110.00 → EUR/JPY = 1.1002 × 110.00 = 121.022
- Since we can sell EUR/JPY at 121.00, which is less than 121.022, no arbitrage here.

- Sell EUR/USD at 1.1000, buy USD/JPY at 110.05 → EUR/JPY = 1.1000 × 110.05 = 121.055  
- Since we can buy EUR/JPY at 121.10, which is more than 121.055, no arbitrage here either.

Actually, let's check the reverse:
- Buy EUR/JPY at 121.00, sell EUR/USD at 1.1000, buy USD/JPY at 110.05
- Effective EUR/JPY rate: 1.1000 × 110.05 = 121.055
- Since 121.00 < 121.055, there's an arbitrage!

**Arbitrage strategy:**
1. Buy €1,000,000 worth of EUR/JPY at 121.00 = ¥121,000,000
2. Sell €1,000,000 for USD at 1.1000 = $1,100,000  
3. Sell $1,100,000 for JPY at 110.05 = ¥121,055,000
4. Profit = ¥121,055,000 - ¥121,000,000 = ¥55,000

**Maximum profit: ¥55,000 per €1,000,000 traded**

---

### Puzzle 2: The Coin Flip Game
**Problem:** You play a game where you flip a fair coin. If heads, you win $1. If tails, you lose $1. You start with $10. What's the probability you'll reach $20 before going broke (reaching $0)?

**Solution:**
This is a classic random walk problem. Let P(i) be the probability of reaching $20 starting from $i.

Boundary conditions:
- P(0) = 0 (already broke)
- P(20) = 1 (already won)

For a fair coin (p = 0.5), the solution is:
P(i) = i/20

Therefore, starting with $10:
**P(10) = 10/20 = 0.5 or 50%**

This makes intuitive sense - with a fair game, your probability of doubling your money equals your current fraction of the target.

---

### Puzzle 3: The Options Market Maker
**Problem:** You're a market maker for a stock option. The stock is at $100, and you're quoting a $100 call option expiring in 1 month. A client wants to buy 1000 contracts. Your theoretical value is $2.50, but you're quoting $2.45 bid, $2.55 ask.

Suddenly, the stock jumps to $102. How do you adjust your quotes, and what's your immediate P&L?

**Solution:**

**Immediate P&L calculation:**
- You're short 1000 call options at $2.55
- Delta of ATM option ≈ 0.5
- Stock move: +$2
- P&L from options position: -1000 × 0.5 × $2 = -$1,000
- P&L from premium collected: +1000 × $2.55 = +$2,550
- Net P&L: +$1,550

**Quote adjustment:**
- New theoretical value ≈ $2.50 + 0.5 × $2 = $3.50
- Adjust quotes wider due to increased risk: $3.40 bid, $3.60 ask
- May need to delta hedge by buying 500 shares of stock

---

### Puzzle 4: The Correlation Trade
**Problem:** You observe that stocks A and B are historically 80% correlated. Today, A is up 5% while B is unchanged. Assuming both stocks have similar volatility, what trading strategy would you implement?

**Solution:**

**Analysis:**
- Historical correlation = 0.8
- Expected move in B given A's 5% move: 0.8 × 5% = 4%
- Current spread: A is 5% ahead of expected relationship

**Trading Strategy:**
1. **Short stock A** (expecting mean reversion)
2. **Long stock B** (expecting catch-up)
3. **Size the trade** based on volatilities to be dollar-neutral

**Risk Management:**
- This is a pairs trade betting on mean reversion
- Risk: correlation could have broken down permanently
- Stop loss if spread widens beyond 2 standard deviations

---

### Puzzle 5: The Probability Paradox
**Problem:** In a deck of 52 cards, what's the probability that the top card and bottom card are the same color?

**Solution:**

**Method 1 - Direct calculation:**
- P(both red) = (26/52) × (25/51) = 25/102
- P(both black) = (26/52) × (25/51) = 25/102  
- P(same color) = 25/102 + 25/102 = 50/102 = **25/51 ≈ 0.4902**

**Method 2 - Conditional probability:**
- Once you see the top card, there are 25 cards of the same color among the remaining 51 cards
- P(same color) = 25/51

---

### Puzzle 6: The High-Frequency Trading Latency
**Problem:** Your HFT system has a latency of 100 microseconds. A competitor's system has 95 microseconds. In a typical trading day with 1000 arbitrage opportunities lasting an average of 200 microseconds each, how many opportunities will you miss due to the latency disadvantage?

**Solution:**

**Analysis:**
- Your latency: 100μs
- Competitor latency: 95μs  
- Latency disadvantage: 5μs
- Opportunity duration: 200μs average

**Calculation:**
If opportunities are uniformly distributed and your competitor always beats you by 5μs:
- You arrive 5μs late to each opportunity
- Remaining duration when you arrive: 200μs - 5μs = 195μs
- Assuming opportunities decay linearly, you capture (195/200) = 97.5% of each opportunity's value

**Opportunities missed entirely:** Very few, since 5μs << 200μs
**Value lost:** Approximately 2.5% of total opportunity value

**Answer: You miss essentially 0 opportunities entirely, but lose about 2.5% of the potential profit from each.**

---

### Puzzle 7: The Market Making Spread
**Problem:** You're market making a stock with the following characteristics:
- Stock price: $50
- Daily volatility: 2%
- Average trade size: 1000 shares
- Your inventory capacity: ±10,000 shares
- Target profit margin: 0.1% per trade

What bid-ask spread should you quote?

**Solution:**

**Components of the spread:**

1. **Order processing cost:** ~0.01% of stock price = $0.005
2. **Inventory risk:** Daily volatility × sqrt(holding period)
   - Assuming 1-hour average holding: 2% × sqrt(1/6.5) ≈ 0.78%
3. **Adverse selection cost:** ~0.05% for liquid stocks
4. **Target profit margin:** 0.1%

**Total spread:** (0.01% + 0.78% + 0.05% + 0.1%) × $50 = 0.94% × $50 = **$0.47**

**Bid-ask spread: $49.77 - $50.23 (approximately $0.46 spread)**

---

### Puzzle 8: The Volatility Smile
**Problem:** You observe the following implied volatilities for 1-month options on a stock trading at $100:

| Strike | Call IV |
|--------|---------|
| $90    | 22%     |
| $95    | 20%     |
| $100   | 18%     |
| $105   | 19%     |
| $110   | 21%     |

Explain this pattern and describe a trading strategy to profit from it.

**Solution:**

**Pattern Analysis:**
This is a classic "volatility smile" showing:
- Higher implied volatility for OTM puts ($90 strike)
- Lower implied volatility for ATM options ($100 strike)  
- Increasing implied volatility for OTM calls ($110 strike)

**Economic Explanation:**
- **Put skew:** Demand for downside protection (crash insurance)
- **Call skew:** Demand for upside exposure with limited downside

**Trading Strategy - Volatility Arbitrage:**
1. **Sell high IV options:** Short $90 puts and $110 calls
2. **Buy low IV options:** Long $100 ATM straddle
3. **Delta hedge:** Maintain market-neutral exposure
4. **Profit source:** Time decay and volatility mean reversion

**Risk:** Realized volatility exceeds the average implied volatility of your net position.

---

### Puzzle 9: The Kelly Criterion
**Problem:** You have a trading strategy with a 60% win rate. When you win, you make 50% profit. When you lose, you lose 40%. What fraction of your capital should you bet on each trade?

**Solution:**

**Kelly Criterion Formula:**
f* = (bp - q) / b

Where:
- b = odds received on a win = 0.5
- p = probability of winning = 0.6
- q = probability of losing = 0.4

**Calculation:**
f* = (0.5 × 0.6 - 0.4) / 0.5
f* = (0.3 - 0.4) / 0.5  
f* = -0.1 / 0.5 = -0.2

**Result:** Kelly criterion gives -20%, meaning **this is a negative expectation bet and you shouldn't play it at all!**

**Verification:**
Expected value = 0.6 × 0.5 + 0.4 × (-0.4) = 0.3 - 0.16 = 0.14 or 14%

Wait, let me recalculate this correctly:
- Win: +50% with probability 60%
- Lose: -40% with probability 40%
- Expected return = 0.6 × 50% + 0.4 × (-40%) = 30% - 16% = 14% > 0

The Kelly formula for this case:
f* = (0.6 × 1.5 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.5 / 1.5 = **33.33%**

**Answer: Bet 33.33% of your capital on each trade.**

---

### Puzzle 10: The Birthday Problem in Trading
**Problem:** In a trading floor with 23 traders, what's the probability that at least two traders have the same birthday? How does this relate to hash collisions in trading systems?

**Solution:**

**Birthday Problem:**
P(at least one match) = 1 - P(no matches)

P(no matches) = 365/365 × 364/365 × 363/365 × ... × 343/365

For 23 people:
P(no matches) ≈ 0.493
P(at least one match) = 1 - 0.493 = **50.7%**

**Trading System Application:**
In trading systems, this relates to:
1. **Order ID collisions:** With random order IDs, collision probability grows quadratically
2. **Hash table design:** Need to size hash tables to minimize collisions  
3. **Risk management:** Position reconciliation across multiple systems
4. **Market data:** Duplicate trade detection algorithms

**Practical Example:**
If your system generates 1 million random order IDs per day using 32-bit integers:
- Birthday paradox suggests collisions become likely around sqrt(2^32) ≈ 65,536 orders
- Much less than 1 million, so you need collision detection!

---

## Interview Tips for Plutus Puzzles

### What Interviewers Look For:
1. **Structured thinking:** Break problems into components
2. **Mathematical rigor:** Show your work, state assumptions
3. **Practical application:** Connect puzzles to real trading scenarios
4. **Risk awareness:** Understand limitations and edge cases
5. **Communication:** Explain reasoning clearly

### Common Puzzle Categories:
- **Probability and statistics**
- **Game theory and optimal strategies**  
- **Financial mathematics**
- **Logic and lateral thinking**
- **System design trade-offs**

### Preparation Strategy:
1. Practice mental math and probability calculations
2. Study classic trading scenarios and market microstructure
3. Review statistics, especially conditional probability
4. Understand options pricing and risk management basics
5. Think about how puzzles apply to real trading problems

Remember: The goal isn't just to solve the puzzle, but to demonstrate how you think about complex problems under pressure - a crucial skill in high-frequency trading environments!
