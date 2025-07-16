# JPMorgan HFT Trading Puzzles and Brain Teasers
==================================================

This collection contains challenging puzzles commonly asked in JPMorgan HFT and quantitative trading interviews. These problems test mathematical reasoning, market intuition, and problem-solving skills essential for high-frequency trading roles.

## Puzzle 1: The Market Making Dilemma

### Problem
You're a market maker for a stock currently trading at $100. You quote a bid of $99.95 and an ask of $100.05. A large institutional client calls and says: "I want to trade 100,000 shares, but I won't tell you whether I'm buying or selling until you give me a price."

What price should you quote, and why?

### Solution & Analysis

**Key Insight**: This is an adverse selection problem. The client has private information about their trading direction, which puts you at a disadvantage.

**Strategic Considerations**:
1. **Information Asymmetry**: The client knows their direction; you don't
2. **Size Impact**: 100,000 shares is likely a significant order
3. **Adverse Selection Cost**: You're more likely to be picked off

**Optimal Response**:
Quote the mid-price ($100.00) but widen your spread significantly.

**Reasoning**:
- At mid-price, you're not biased toward either direction
- Wider spread compensates for adverse selection risk
- Alternative: Decline the trade or request more information

**Mathematical Framework**:
If P(buy) = P(sell) = 0.5, but the client has edge e:
- Your expected P&L = -e × position × client_edge
- Required spread = 2 × e × client_edge

---

## Puzzle 2: The Coin Toss Paradox

### Problem
You have a biased coin that lands heads with probability p. You flip it repeatedly and stop when you see either HH (two heads in a row) or HT (heads followed by tails). What's the probability that you stop on HH?

### Solution

**State-Based Analysis**:

Define states:
- S: Start state
- H: Just saw heads
- HH: Saw two heads (terminal)
- HT: Saw heads then tails (terminal)

**Transition Probabilities**:
- From S: probability p to H, probability (1-p) stay in S
- From H: probability p to HH, probability (1-p) to HT

**System of Equations**:
Let P(S) = probability of reaching HH from start
Let P(H) = probability of reaching HH after seeing one head

P(S) = p × P(H) + (1-p) × P(S)
P(H) = p × 1 + (1-p) × 0 = p

**Solving**:
P(S) = p × p + (1-p) × P(S)
P(S) = p² + (1-p) × P(S)
P(S) × p = p²
P(S) = p

**Answer**: The probability is p (same as the coin's bias).

**Intuitive Explanation**: Since both sequences start with H, and that's the "bottleneck," the probability equals the probability of getting the first heads.

---

## Puzzle 3: The Options Arbitrage

### Problem
You observe the following option prices for a stock trading at $100:
- Call with strike $95: $8
- Call with strike $100: $4  
- Call with strike $105: $3

Can you create a risk-free profit? If so, how much?

### Solution

**Arbitrage Check**: Examine convexity condition for call options.

For strikes K₁ < K₂ < K₃, call prices C₁, C₂, C₃ must satisfy:
C₂ ≤ (K₃ - K₂)/(K₃ - K₁) × C₁ + (K₂ - K₁)/(K₃ - K₁) × C₃

**Given Data**:
- K₁ = $95, C₁ = $8
- K₂ = $100, C₂ = $4  
- K₃ = $105, C₃ = $3

**Convexity Check**:
Required: C₂ ≤ (5/10) × 8 + (5/10) × 3 = 4 + 1.5 = 5.5

Actual: C₂ = $4

Since $4 < $5.5, no arbitrage exists from convexity violation.

**Call Spread Arbitrage Check**:
- Buy $95 call, sell $100 call: Net cost = $8 - $4 = $4
- Payoff at expiration: max(S - 95, 0) - max(S - 100, 0)
- This is worth at most $5 (when S ≥ $100)

Since we pay $4 for something worth at most $5, and can be worth $0, this is not arbitrage.

**Butterfly Spread Check**:
Buy 1 call at $95, buy 1 call at $105, sell 2 calls at $100:
Cost = $8 + $3 - 2×$4 = $3

Payoff function:
- S ≤ $95: $0
- $95 < S ≤ $100: S - $95
- $100 < S ≤ $105: $205 - 2S
- S > $105: $0

Maximum payoff = $5 (at S = $100)

Since we pay $3 for minimum $0 and maximum $5, this could be profitable but isn't risk-free arbitrage.

**Conclusion**: No pure arbitrage exists with these prices.

---

## Puzzle 4: The Random Walk Race

### Problem
Two particles start at position 0 on a number line. Each second, each particle moves +1 or -1 with equal probability, independently. What's the probability that they are ever at the same position again?

### Solution

**Relative Motion Analysis**:
Define the difference D(t) = X₁(t) - X₂(t) where X₁, X₂ are particle positions.

**Key Insight**: D(t) is also a random walk starting at D(0) = 0.

At each step: D(t+1) = D(t) + ΔD where ΔD ∈ {-2, 0, +2}
- P(ΔD = -2) = 1/4 (both particles move in opposite directions)
- P(ΔD = 0) = 1/2 (both move in same direction)  
- P(ΔD = +2) = 1/4

**Equivalent Problem**: What's the probability that a random walk starting at 0 returns to 0?

**Classical Result**: For a symmetric random walk in 1D, the probability of eventual return to the starting point is 1.

**Verification**: Using generating functions or the reflection principle confirms this result.

**Answer**: Probability = 1 (they will definitely meet again).

**Expected Time**: The expected time until they meet again is infinite, even though they meet with probability 1.

---

## Puzzle 5: The Correlation Trade

### Problem
You have two assets whose daily returns have correlation ρ = 0.8. Today, Asset A moved up 2% while Asset B moved down 1%. Based on this information alone, what's your best estimate for how much Asset B should have moved?

### Solution

**Statistical Framework**:
Let r_A and r_B be the daily returns. Given:
- Correlation(r_A, r_B) = 0.8
- Today: r_A = 2%, r_B = -1%

**Linear Regression Approach**:
The best linear predictor of r_B given r_A is:
E[r_B | r_A] = ρ × (σ_B/σ_A) × r_A

**Assumptions Needed**:
Without additional information, assume σ_A = σ_B (equal volatilities).

**Calculation**:
E[r_B | r_A = 2%] = 0.8 × 1 × 2% = 1.6%

**Interpretation**:
Given the 2% move in Asset A, we would have expected Asset B to move +1.6%.
Actual move was -1%, so Asset B underperformed expectation by 2.6%.

**Trading Implication**:
This suggests a potential mean-reversion opportunity:
- Asset B might be temporarily undervalued relative to Asset A
- Could consider buying B, selling A (pairs trade)

**Risk Considerations**:
- Correlation might be time-varying
- Single observation has high noise
- External factors might have affected Asset B

---

## Puzzle 6: The Latency Arbitrage

### Problem
Your HFT system has a 0.1ms latency advantage over competitors. A news event just broke that will move stock XYZ by exactly $1 (either up or down with equal probability). The current bid is $99.50, ask is $100.50. How much profit can you expect to make?

### Solution

**Information Advantage Analysis**:

**Scenario Analysis**:
1. **Positive News** (50% probability):
   - Stock should trade at ~$101.50
   - You can buy at $100.50 (current ask) before others react
   - Expected profit per share: $101.50 - $100.50 = $1.00

2. **Negative News** (50% probability):
   - Stock should trade at ~$98.50  
   - You can sell at $99.50 (current bid) before others react
   - Expected profit per share: $99.50 - $98.50 = $1.00

**Position Sizing Constraint**:
Your advantage only lasts 0.1ms, so you're limited by:
- Available liquidity at current bid/ask
- Your capital and risk limits
- Market impact of your trades

**Expected Profit Calculation**:
E[Profit per share] = 0.5 × $1.00 + 0.5 × $1.00 = $1.00

**Practical Considerations**:
- Liquidity may be limited (maybe only 100-1000 shares available)
- Need to execute before 0.1ms advantage expires
- Transaction costs reduce profit
- Risk of partial fills

**Realistic Estimate**:
If you can trade 500 shares: Expected profit = 500 × $1.00 = $500
Less transaction costs (~$2): Net profit ≈ $498

---

## Puzzle 7: The Volatility Puzzle

### Problem
You notice that implied volatility for at-the-money options is 20%, but you estimate the actual volatility will be 25%. How do you profit from this discrepancy, and what's your expected return?

### Solution

**Volatility Arbitrage Strategy**:

**Step 1: Analysis**
- Market implies 20% volatility
- You believe actual will be 25%
- Options are underpriced if your estimate is correct

**Step 2: Strategy (Long Volatility)**
1. **Buy the option** (underpriced)
2. **Delta hedge** with underlying stock
3. **Rebalance hedge** as needed
4. **Profit from gamma scalping**

**Step 3: P&L Analysis**

**Theoretical Edge**:
For ATM option with time T to expiration:
Edge ≈ 0.5 × Gamma × S² × (σ²_actual - σ²_implied) × T

**Numerical Example**:
- Stock price S = $100
- Time to expiration T = 0.25 years (3 months)
- Gamma ≈ 0.01 (typical for ATM option)

Edge ≈ 0.5 × 0.01 × 100² × (0.25² - 0.20²) × 0.25
Edge ≈ 0.5 × 0.01 × 10,000 × (0.0625 - 0.04) × 0.25
Edge ≈ 0.5 × 0.01 × 10,000 × 0.0225 × 0.25
Edge ≈ $2.81 per option

**Risk Considerations**:
- Volatility estimate might be wrong
- Transaction costs from rebalancing
- Model risk (Black-Scholes assumptions)
- Liquidity risk in underlying

**Expected Return**:
If option costs $4 and edge is $2.81:
Expected return = $2.81 / $4 = 70%

---

## Puzzle 8: The Probability Chain

### Problem
In a certain market, the probability of an up move tomorrow depends on today's move:
- If today was up: P(tomorrow up) = 0.6
- If today was down: P(tomorrow up) = 0.4

Starting from an up day, what's the long-run probability of being in an up state?

### Solution

**Markov Chain Analysis**:

**Transition Matrix**:
```
       Tomorrow
Today    Up   Down
Up     [0.6   0.4]
Down   [0.4   0.6]
```

**Steady-State Calculation**:
Let π = [π_up, π_down] be steady-state probabilities.

From π = πP:
π_up = 0.6 × π_up + 0.4 × π_down
π_down = 0.4 × π_up + 0.6 × π_down

Also: π_up + π_down = 1

**Solving**:
π_up = 0.6 × π_up + 0.4 × (1 - π_up)
π_up = 0.6 × π_up + 0.4 - 0.4 × π_up
π_up = 0.2 × π_up + 0.4
0.8 × π_up = 0.4
π_up = 0.5

**Answer**: Long-run probability of up state = 50%

**Insight**: Despite the momentum effect, the system converges to equal probabilities due to symmetry in the transition matrix.

---

## Puzzle 9: The Order Book Mystery

### Problem
You're watching an order book where the best bid is $99.95 (size 1000) and best ask is $100.05 (size 800). A market buy order for 1200 shares arrives. What will be the volume-weighted average price (VWAP) of this trade?

### Solution

**Order Execution Analysis**:

**Available Liquidity**:
- Ask level 1: $100.05 with 800 shares
- Ask level 2: Unknown (need assumption)

**Assumption**: Next ask level is $100.15 with sufficient size.

**Trade Execution**:
1. First 800 shares execute at $100.05
2. Remaining 400 shares execute at $100.15

**VWAP Calculation**:
VWAP = (800 × $100.05 + 400 × $100.15) / 1200
VWAP = ($80,040 + $40,060) / 1200
VWAP = $120,100 / 1200
VWAP = $100.083

**Market Impact**:
- Average execution price: $100.083
- Mid-price before trade: $100.00
- Market impact: $0.083 per share = 8.3 cents

**Sensitivity Analysis**:
If second level was at $100.10:
VWAP = (800 × $100.05 + 400 × $100.10) / 1200 = $100.067

---

## Puzzle 10: The Statistical Arbitrage Opportunity

### Problem
You've identified that the spread between two stocks follows a mean-reverting process. Currently, the spread is 2 standard deviations above its mean. Historically, when the spread reaches this level, it reverts to within 1 standard deviation of the mean 80% of the time within 5 days. However, 20% of the time, it continues to diverge further.

If the potential profit from mean reversion is $10,000 and the potential loss from divergence is $15,000, should you take this trade?

### Solution

**Expected Value Analysis**:

**Trade Setup**:
- Probability of mean reversion: 80%
- Probability of further divergence: 20%
- Profit if correct: $10,000
- Loss if wrong: $15,000

**Expected Value**:
E[P&L] = 0.8 × $10,000 + 0.2 × (-$15,000)
E[P&L] = $8,000 - $3,000
E[P&L] = $5,000

**Decision**: Yes, take the trade (positive expected value).

**Risk Considerations**:
1. **Kelly Criterion**: Optimal position sizing
   - Win probability: 80%
   - Win amount: $10,000
   - Loss amount: $15,000
   - Kelly fraction: (0.8 × 10,000 - 0.2 × 15,000) / 15,000 = 33%

2. **Risk of Ruin**: With 20% chance of losing $15,000, ensure adequate capital

3. **Model Risk**: Historical patterns might not persist

4. **Liquidity Risk**: Ability to exit position if needed

**Enhanced Strategy**:
- Use stop-losses to limit downside
- Scale into position gradually
- Monitor correlation breakdown
- Set position size based on Kelly criterion

---

## Summary: Key Skills for JPMorgan HFT Interviews

### Mathematical Concepts:
1. **Probability Theory**: Conditional probability, Bayes' theorem
2. **Stochastic Processes**: Random walks, Markov chains
3. **Statistics**: Correlation, regression, expected values
4. **Optimization**: Kelly criterion, portfolio theory

### Market Knowledge:
1. **Options**: Arbitrage bounds, Greeks, volatility
2. **Market Microstructure**: Order books, market impact
3. **Risk Management**: VaR, correlation, diversification
4. **Trading Strategies**: Statistical arbitrage, market making

### Problem-Solving Approach:
1. **Identify the core issue** (arbitrage, optimization, probability)
2. **State assumptions clearly**
3. **Use multiple solution methods** when possible
4. **Consider practical constraints**
5. **Discuss risk and implementation**

### Interview Tips:
- Think out loud and explain your reasoning
- Ask clarifying questions about assumptions
- Consider edge cases and practical limitations
- Connect solutions to real trading scenarios
- Demonstrate both theoretical knowledge and practical insight

These puzzles represent the type of analytical thinking valued at JPMorgan's quantitative trading desks, combining mathematical rigor with market intuition.
