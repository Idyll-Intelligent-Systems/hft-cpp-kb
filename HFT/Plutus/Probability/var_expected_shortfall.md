# Plutus HFT Probability Problem: Risk Management and VAR Calculations

## Problem Statement

You are a quantitative analyst at Plutus, a high-frequency trading firm. Your portfolio consists of various financial instruments, and you need to calculate the Value at Risk (VaR) and Expected Shortfall (ES) under different scenarios.

**Given:**
- Portfolio consists of 3 assets with the following daily returns characteristics:
  - Asset A: μ = 0.08% daily, σ = 1.2% daily
  - Asset B: μ = 0.05% daily, σ = 0.8% daily  
  - Asset C: μ = 0.12% daily, σ = 2.1% daily
- Portfolio weights: [0.4, 0.3, 0.3]
- Correlation matrix:
  ```
       A     B     C
  A  1.00  0.25  0.15
  B  0.25  1.00  0.35
  C  0.15  0.35  1.00
  ```
- Portfolio value: $10,000,000
- Confidence levels: 95%, 99%, 99.9%

## Questions

### 1. Portfolio Statistics Calculation
Calculate the portfolio's expected daily return and daily volatility.

**Solution:**

**Expected Portfolio Return:**
μ_p = w₁μ₁ + w₂μ₂ + w₃μ₃
μ_p = 0.4(0.0008) + 0.3(0.0005) + 0.3(0.0012)
μ_p = 0.00032 + 0.00015 + 0.00036 = **0.00083 or 0.083%**

**Portfolio Variance:**
σ²_p = Σᵢ Σⱼ wᵢwⱼσᵢσⱼρᵢⱼ

Expanding:
- w₁²σ₁² = (0.4)²(0.012)² = 0.000023040
- w₂²σ₂² = (0.3)²(0.008)² = 0.000005760  
- w₃²σ₃² = (0.3)²(0.021)² = 0.000039690
- 2w₁w₂σ₁σ₂ρ₁₂ = 2(0.4)(0.3)(0.012)(0.008)(0.25) = 0.000005760
- 2w₁w₃σ₁σ₃ρ₁₃ = 2(0.4)(0.3)(0.012)(0.021)(0.15) = 0.000004536
- 2w₂w₃σ₂σ₃ρ₂₃ = 2(0.3)(0.3)(0.008)(0.021)(0.35) = 0.000005292

σ²_p = 0.000023040 + 0.000005760 + 0.000039690 + 0.000005760 + 0.000004536 + 0.000005292
σ²_p = 0.000084078

**Portfolio Daily Volatility:**
σ_p = √0.000084078 = **0.00917 or 0.917%**

### 2. Value at Risk (VaR) Calculation

**Parametric VaR (Normal Distribution Assumption):**

VaR = Portfolio Value × |μ_p - z_α × σ_p|

Where z_α is the critical value for confidence level α:
- 95%: z₀.₀₅ = 1.645
- 99%: z₀.₀₁ = 2.326  
- 99.9%: z₀.₀₀₁ = 3.090

**Calculations:**

**95% VaR:**
VaR₉₅% = $10,000,000 × |0.00083 - 1.645 × 0.00917|
VaR₉₅% = $10,000,000 × |0.00083 - 0.01508|
VaR₉₅% = $10,000,000 × 0.01425 = **$142,500**

**99% VaR:**
VaR₉₉% = $10,000,000 × |0.00083 - 2.326 × 0.00917|
VaR₉₉% = $10,000,000 × |0.00083 - 0.02133|
VaR₉₉% = $10,000,000 × 0.02050 = **$205,000**

**99.9% VaR:**
VaR₉₉.₉% = $10,000,000 × |0.00083 - 3.090 × 0.00917|
VaR₉₉.₉% = $10,000,000 × |0.00083 - 0.02833|
VaR₉₉.₉% = $10,000,000 × 0.02750 = **$275,000**

### 3. Expected Shortfall (ES) Calculation

Expected Shortfall represents the expected loss given that VaR is exceeded.

**Formula:**
ES_α = Portfolio Value × [μ_p - σ_p × φ(z_α)/(1-α)]

Where φ(z) is the standard normal PDF.

**Calculations:**

**95% ES:**
φ(1.645) = 0.1031
ES₉₅% = $10,000,000 × [0.00083 - 0.00917 × 0.1031/0.05]
ES₉₅% = $10,000,000 × [0.00083 - 0.01892]
ES₉₅% = $10,000,000 × 0.01809 = **$180,900**

**99% ES:**
φ(2.326) = 0.0267  
ES₉₉% = $10,000,000 × [0.00083 - 0.00917 × 0.0267/0.01]
ES₉₉% = $10,000,000 × [0.00083 - 0.02449]
ES₉₉% = $10,000,000 × 0.02366 = **$236,600**

**99.9% ES:**
φ(3.090) = 0.0017
ES₉₉.₉% = $10,000,000 × [0.00083 - 0.00917 × 0.0017/0.001]
ES₉₉.₉% = $10,000,000 × [0.00083 - 0.01559]
ES₉₉.₉% = $10,000,000 × 0.01476 = **$147,600**

### 4. Monte Carlo Simulation Approach

For validation, we can use Monte Carlo simulation:

**Algorithm:**
1. Generate correlated random variables using Cholesky decomposition
2. Simulate portfolio returns for N scenarios (e.g., 100,000)
3. Calculate empirical VaR and ES from the distribution

**Python Implementation Outline:**
```python
import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky

def monte_carlo_var_es(correlation_matrix, means, volatilities, weights, 
                       portfolio_value, confidence_levels, num_simulations=100000):
    
    # Cholesky decomposition for correlation
    L = cholesky(correlation_matrix, lower=True)
    
    # Generate independent random variables
    Z = np.random.standard_normal((num_simulations, len(means)))
    
    # Transform to correlated variables
    correlated_returns = Z @ L.T
    
    # Scale by volatility and add mean
    asset_returns = means + correlated_returns * volatilities
    
    # Calculate portfolio returns
    portfolio_returns = asset_returns @ weights
    
    # Calculate portfolio P&L
    portfolio_pnl = portfolio_value * portfolio_returns
    
    # Calculate VaR and ES
    results = {}
    for conf_level in confidence_levels:
        alpha = 1 - conf_level/100
        var_quantile = np.percentile(portfolio_pnl, alpha*100)
        es = np.mean(portfolio_pnl[portfolio_pnl <= var_quantile])
        
        results[conf_level] = {
            'VaR': -var_quantile,  # Convert to positive loss
            'ES': -es
        }
    
    return results
```

### 5. Risk Management Implications

**Key Insights:**

1. **Diversification Benefit:** The portfolio volatility (0.917%) is less than the weighted average of individual volatilities (1.41%), demonstrating diversification benefits.

2. **Tail Risk:** Expected Shortfall is consistently higher than VaR, showing that losses beyond VaR threshold can be significantly larger.

3. **Confidence Level Impact:** Moving from 95% to 99.9% confidence increases VaR by 93%, highlighting tail risk sensitivity.

4. **Risk Budgeting:** Asset C contributes most to portfolio risk despite only 30% weight due to its high volatility.

### 6. Extensions and Real-World Considerations

**Advanced Topics for Plutus Interview:**

1. **Non-Normal Distributions:**
   - Fat tails and skewness in financial returns
   - Student's t-distribution or skewed-t models
   - Extreme Value Theory for tail modeling

2. **Dynamic Risk Models:**
   - GARCH models for time-varying volatility
   - DCC-GARCH for dynamic correlations
   - Regime-switching models

3. **Backtesting:**
   - Kupiec POF test for VaR validation
   - Christoffersen independence test
   - ES backtesting using scoring functions

4. **Regulatory Capital:**
   - Basel III market risk requirements
   - Fundamental Review of Trading Book (FRTB)
   - Model validation and governance

5. **High-Frequency Considerations:**
   - Intraday VaR calculations
   - Microstructure noise in correlation estimates
   - Flash crash scenarios and liquidity risk

### 7. Interview Questions

**Typical Plutus Probability Questions:**

1. **Q:** What are the limitations of normal distribution assumption in VaR?
   **A:** Fat tails, skewness, volatility clustering, and correlation breakdown during crises.

2. **Q:** How would you handle correlation breakdown during market stress?
   **A:** Use stressed correlation matrices, DCC models, or copula-based approaches.

3. **Q:** Explain the difference between VaR and Expected Shortfall.
   **A:** VaR is a quantile, ES is the expected loss beyond VaR. ES is coherent, VaR is not sub-additive.

4. **Q:** How do you validate a VaR model?
   **A:** Backtesting with violation rates, independence tests, and profit/loss attribution.

5. **Q:** What's the impact of non-synchronous trading on correlation estimates?
   **A:** Spurious correlation reduction due to stale prices; use lead-lag models or overnight returns.

## Summary

This problem demonstrates essential risk management concepts for HFT firms:
- Portfolio theory and diversification
- Statistical risk measures (VaR, ES)
- Monte Carlo simulation techniques
- Real-world modeling considerations
- Regulatory and business implications

Understanding these concepts is crucial for quantitative roles at firms like Plutus, where rapid decision-making and accurate risk assessment are paramount.
