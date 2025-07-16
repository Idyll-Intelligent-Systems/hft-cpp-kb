"""
Plutus HFT Mathematics Problem: Options Pricing Models Implementation (Python)
=============================================================================

Problem Statement:
Implement Black-Scholes option pricing model, Binomial model, and Monte Carlo simulation
for option pricing. Compare results and analyze Greeks.

This Python implementation provides:
1. Cleaner mathematical notation using NumPy
2. Better data visualization capabilities
3. Integration with pandas for data analysis
4. Easier statistical computations

Applications:
- Rapid prototyping of pricing models
- Data analysis and backtesting
- Visualization of Greeks and price surfaces
- Research and model validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BlackScholesResult:
    """Data class to store Black-Scholes results"""
    call_price: float
    put_price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

@dataclass
class MonteCarloResult:
    """Data class to store Monte Carlo results"""
    price: float
    standard_error: float
    confidence_interval: Tuple[float, float]
    paths: Optional[np.ndarray] = None

class OptionsPricingModels:
    """
    Comprehensive options pricing models implementation
    """
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility"""
        np.random.seed(seed)
    
    def black_scholes(self, S: float, K: float, T: float, r: float, 
                     sigma: float, option_type: str = 'call') -> BlackScholesResult:
        """
        Black-Scholes option pricing model
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility
        option_type: 'call' or 'put'
        
        Returns:
        BlackScholesResult with price and Greeks
        """
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Standard normal CDF values
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)
        
        # Standard normal PDF value
        n_d1 = norm.pdf(d1)
        
        # Option prices
        call_price = S * N_d1 - K * np.exp(-r * T) * N_d2
        put_price = K * np.exp(-r * T) * N_neg_d2 - S * N_neg_d1
        
        # Greeks
        delta_call = N_d1
        delta_put = N_d1 - 1
        
        gamma = n_d1 / (S * sigma * np.sqrt(T))
        
        theta_call = (-S * n_d1 * sigma / (2 * np.sqrt(T)) - 
                     r * K * np.exp(-r * T) * N_d2) / 365
        theta_put = (-S * n_d1 * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * N_neg_d2) / 365
        
        vega = S * n_d1 * np.sqrt(T) / 100
        
        rho_call = K * T * np.exp(-r * T) * N_d2 / 100
        rho_put = -K * T * np.exp(-r * T) * N_neg_d2 / 100
        
        if option_type.lower() == 'call':
            return BlackScholesResult(
                call_price=call_price,
                put_price=put_price,
                delta=delta_call,
                gamma=gamma,
                theta=theta_call,
                vega=vega,
                rho=rho_call
            )
        else:
            return BlackScholesResult(
                call_price=call_price,
                put_price=put_price,
                delta=delta_put,
                gamma=gamma,
                theta=theta_put,
                vega=vega,
                rho=rho_put
            )
    
    def binomial_option_price(self, S: float, K: float, T: float, r: float, 
                             sigma: float, steps: int, option_type: str = 'call',
                             american: bool = False) -> float:
        """
        Binomial option pricing model
        
        Parameters:
        american: True for American option, False for European
        """
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize stock prices at maturity
        stock_prices = np.zeros(steps + 1)
        for i in range(steps + 1):
            stock_prices[i] = S * (u ** (steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        if option_type.lower() == 'call':
            option_values = np.maximum(stock_prices - K, 0)
        else:
            option_values = np.maximum(K - stock_prices, 0)
        
        # Work backwards through the tree
        for j in range(steps - 1, -1, -1):
            for i in range(j + 1):
                # European option value
                option_values[i] = np.exp(-r * dt) * (
                    p * option_values[i] + (1 - p) * option_values[i + 1]
                )
                
                # American option early exercise
                if american:
                    stock_price = S * (u ** (j - i)) * (d ** i)
                    if option_type.lower() == 'call':
                        intrinsic_value = max(stock_price - K, 0)
                    else:
                        intrinsic_value = max(K - stock_price, 0)
                    option_values[i] = max(option_values[i], intrinsic_value)
        
        return option_values[0]
    
    def monte_carlo_option_price(self, S: float, K: float, T: float, r: float,
                                sigma: float, num_simulations: int,
                                option_type: str = 'call',
                                confidence_level: float = 0.95,
                                save_paths: bool = False) -> MonteCarloResult:
        """
        Monte Carlo option pricing simulation
        """
        # Generate random numbers
        Z = np.random.standard_normal(num_simulations)
        
        # Calculate terminal stock prices
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        # Calculate standard error
        standard_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(num_simulations)
        
        # Confidence interval
        z_score = norm.ppf((1 + confidence_level) / 2)
        ci_lower = option_price - z_score * standard_error
        ci_upper = option_price + z_score * standard_error
        
        paths = ST if save_paths else None
        
        return MonteCarloResult(
            price=option_price,
            standard_error=standard_error,
            confidence_interval=(ci_lower, ci_upper),
            paths=paths
        )
    
    def asian_option_price(self, S: float, K: float, T: float, r: float,
                          sigma: float, num_simulations: int, num_steps: int,
                          option_type: str = 'call', average_type: str = 'arithmetic') -> float:
        """
        Asian option pricing using Monte Carlo
        
        Parameters:
        average_type: 'arithmetic' or 'geometric'
        """
        dt = T / num_steps
        
        # Initialize arrays
        payoffs = np.zeros(num_simulations)
        
        for i in range(num_simulations):
            # Generate path
            path = np.zeros(num_steps + 1)
            path[0] = S
            
            for j in range(1, num_steps + 1):
                Z = np.random.standard_normal()
                path[j] = path[j-1] * np.exp((r - 0.5 * sigma**2) * dt + 
                                           sigma * np.sqrt(dt) * Z)
            
            # Calculate average
            if average_type == 'arithmetic':
                average_price = np.mean(path[1:])  # Exclude initial price
            else:  # geometric
                average_price = np.exp(np.mean(np.log(path[1:])))
            
            # Calculate payoff
            if option_type.lower() == 'call':
                payoffs[i] = max(average_price - K, 0)
            else:
                payoffs[i] = max(K - average_price, 0)
        
        return np.exp(-r * T) * np.mean(payoffs)
    
    def barrier_option_price(self, S: float, K: float, T: float, r: float,
                           sigma: float, barrier: float, num_simulations: int,
                           num_steps: int, option_type: str = 'call',
                           barrier_type: str = 'up-and-out') -> float:
        """
        Barrier option pricing using Monte Carlo
        
        Parameters:
        barrier_type: 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
        """
        dt = T / num_steps
        payoffs = np.zeros(num_simulations)
        
        for i in range(num_simulations):
            # Generate path
            path = np.zeros(num_steps + 1)
            path[0] = S
            barrier_hit = False
            
            for j in range(1, num_steps + 1):
                Z = np.random.standard_normal()
                path[j] = path[j-1] * np.exp((r - 0.5 * sigma**2) * dt + 
                                           sigma * np.sqrt(dt) * Z)
                
                # Check barrier condition
                if 'up' in barrier_type and path[j] >= barrier:
                    barrier_hit = True
                elif 'down' in barrier_type and path[j] <= barrier:
                    barrier_hit = True
            
            # Calculate payoff based on barrier type
            if option_type.lower() == 'call':
                intrinsic_value = max(path[-1] - K, 0)
            else:
                intrinsic_value = max(K - path[-1], 0)
            
            if 'out' in barrier_type:
                payoffs[i] = 0 if barrier_hit else intrinsic_value
            else:  # 'in' type
                payoffs[i] = intrinsic_value if barrier_hit else 0
        
        return np.exp(-r * T) * np.mean(payoffs)
    
    def implied_volatility(self, market_price: float, S: float, K: float,
                          T: float, r: float, option_type: str = 'call',
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Brent's method
        """
        def objective(sigma):
            try:
                bs_result = self.black_scholes(S, K, T, r, sigma, option_type)
                model_price = bs_result.call_price if option_type.lower() == 'call' else bs_result.put_price
                return model_price - market_price
            except:
                return float('inf')
        
        try:
            # Use Brent's method for root finding
            implied_vol = brentq(objective, 0.001, 5.0, maxiter=max_iterations, xtol=tolerance)
            return implied_vol
        except:
            # Fallback to Newton-Raphson if Brent's method fails
            sigma = 0.2  # Initial guess
            for _ in range(max_iterations):
                bs_result = self.black_scholes(S, K, T, r, sigma, option_type)
                model_price = bs_result.call_price if option_type.lower() == 'call' else bs_result.put_price
                vega = bs_result.vega * 100  # Convert from percentage
                
                price_diff = model_price - market_price
                if abs(price_diff) < tolerance:
                    return sigma
                
                if abs(vega) < 1e-10:
                    break
                
                sigma = sigma - price_diff / vega
                sigma = max(sigma, 0.001)  # Keep positive
            
            return sigma
    
    def volatility_surface(self, S: float, strikes: List[float], 
                          maturities: List[float], r: float,
                          market_prices: np.ndarray) -> pd.DataFrame:
        """
        Calculate implied volatility surface
        
        Parameters:
        market_prices: 2D array with shape (len(maturities), len(strikes))
        """
        vol_surface = np.zeros_like(market_prices)
        
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                market_price = market_prices[i, j]
                if market_price > 0:  # Valid market price
                    vol_surface[i, j] = self.implied_volatility(
                        market_price, S, K, T, r, 'call'
                    )
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(vol_surface, index=maturities, columns=strikes)
        df.index.name = 'Maturity'
        df.columns.name = 'Strike'
        
        return df

def run_comprehensive_tests():
    """Run comprehensive tests and analysis"""
    print("Plutus HFT Options Pricing Models - Python Implementation")
    print("=" * 60)
    
    model = OptionsPricingModels()
    
    # Standard parameters
    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.2
    
    print("\n1. Black-Scholes Model Analysis")
    print("-" * 40)
    
    bs_result = model.black_scholes(S, K, T, r, sigma, 'call')
    print(f"Call Price: ${bs_result.call_price:.4f}")
    print(f"Put Price: ${bs_result.put_price:.4f}")
    print(f"Delta: {bs_result.delta:.4f}")
    print(f"Gamma: {bs_result.gamma:.4f}")
    print(f"Theta: {bs_result.theta:.4f}")
    print(f"Vega: {bs_result.vega:.4f}")
    print(f"Rho: {bs_result.rho:.4f}")
    
    # Put-Call Parity verification
    pcp_lhs = bs_result.call_price - bs_result.put_price
    pcp_rhs = S - K * np.exp(-r * T)
    print(f"\nPut-Call Parity Check: {abs(pcp_lhs - pcp_rhs):.10f} (should be ~0)")
    
    print("\n2. Model Comparison")
    print("-" * 40)
    
    # Binomial model convergence
    steps_list = [10, 50, 100, 500, 1000]
    print("Binomial Model Convergence:")
    for steps in steps_list:
        start_time = time.time()
        binomial_price = model.binomial_option_price(S, K, T, r, sigma, steps, 'call')
        end_time = time.time()
        diff = abs(binomial_price - bs_result.call_price)
        print(f"  Steps: {steps:4d}, Price: ${binomial_price:.4f}, "
              f"Diff from BS: ${diff:.4f}, Time: {(end_time-start_time)*1000:.2f}ms")
    
    # Monte Carlo simulation
    print("\nMonte Carlo Simulation:")
    sim_counts = [1000, 10000, 100000, 1000000]
    for num_sims in sim_counts:
        start_time = time.time()
        mc_result = model.monte_carlo_option_price(S, K, T, r, sigma, num_sims, 'call')
        end_time = time.time()
        diff = abs(mc_result.price - bs_result.call_price)
        print(f"  Sims: {num_sims:7d}, Price: ${mc_result.price:.4f} ± {mc_result.standard_error:.4f}, "
              f"Diff: ${diff:.4f}, Time: {(end_time-start_time)*1000:.1f}ms")
    
    print("\n3. Exotic Options Pricing")
    print("-" * 40)
    
    # Asian option
    asian_price = model.asian_option_price(S, K, T, r, sigma, 50000, 252, 'call')
    print(f"Asian Call Option: ${asian_price:.4f}")
    
    # Barrier options
    barrier = 110.0
    knock_out = model.barrier_option_price(S, K, T, r, sigma, barrier, 50000, 252, 'call', 'up-and-out')
    knock_in = model.barrier_option_price(S, K, T, r, sigma, barrier, 50000, 252, 'call', 'up-and-in')
    print(f"Up-and-Out Barrier Option (Barrier=${barrier}): ${knock_out:.4f}")
    print(f"Up-and-In Barrier Option (Barrier=${barrier}): ${knock_in:.4f}")
    print(f"Sum of Barrier Options: ${knock_out + knock_in:.4f}")
    print(f"Vanilla Call: ${bs_result.call_price:.4f}")
    
    print("\n4. Implied Volatility Analysis")
    print("-" * 40)
    
    # Test implied volatility calculation
    market_price = bs_result.call_price
    implied_vol = model.implied_volatility(market_price, S, K, T, r, 'call')
    print(f"Original Volatility: {sigma*100:.2f}%")
    print(f"Market Price: ${market_price:.4f}")
    print(f"Implied Volatility: {implied_vol*100:.2f}%")
    print(f"Difference: {abs(sigma - implied_vol)*100:.6f}%")
    
    print("\n5. Greeks Sensitivity Analysis")
    print("-" * 40)
    
    # Analyze Greeks across different scenarios
    scenarios = [
        ("Current", S, sigma),
        ("Stock +10%", S * 1.1, sigma),
        ("Stock -10%", S * 0.9, sigma),
        ("Vol +5%", S, sigma + 0.05),
        ("Vol -5%", S, sigma - 0.05)
    ]
    
    print(f"{'Scenario':<12} {'Price':<8} {'Delta':<8} {'Gamma':<8} {'Theta':<8} {'Vega':<8}")
    print("-" * 60)
    for name, spot, vol in scenarios:
        result = model.black_scholes(spot, K, T, r, vol, 'call')
        print(f"{name:<12} ${result.call_price:<7.2f} {result.delta:<7.3f} "
              f"{result.gamma:<7.3f} {result.theta:<7.2f} {result.vega:<7.2f}")

def create_volatility_surface_example():
    """Create and display a volatility surface example"""
    print("\n6. Volatility Surface Construction")
    print("-" * 40)
    
    model = OptionsPricingModels()
    S = 100.0
    r = 0.05
    
    # Define strikes and maturities
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    maturities = [0.083, 0.25, 0.5, 1.0]  # 1M, 3M, 6M, 1Y
    
    # Generate synthetic market prices (using different volatilities)
    market_prices = np.zeros((len(maturities), len(strikes)))
    
    # Create volatility smile/skew pattern
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            # Simulate volatility smile (higher vol for OTM options)
            moneyness = K / S
            base_vol = 0.2
            skew = -0.1 * (moneyness - 1)  # Negative skew
            smile = 0.05 * (moneyness - 1)**2  # Smile curvature
            term_structure = 0.02 * np.sqrt(T)  # Term structure effect
            
            vol = base_vol + skew + smile + term_structure
            vol = max(vol, 0.05)  # Minimum volatility
            
            # Calculate theoretical price
            bs_result = model.black_scholes(S, K, T, r, vol, 'call')
            market_prices[i, j] = bs_result.call_price
    
    # Calculate implied volatility surface
    vol_surface = model.volatility_surface(S, strikes, maturities, r, market_prices)
    
    print("\nImplied Volatility Surface (%):")
    print(vol_surface.round(4) * 100)

if __name__ == "__main__":
    run_comprehensive_tests()
    create_volatility_surface_example()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("\nKey Takeaways for Plutus Interview:")
    print("1. Black-Scholes provides analytical solution for European options")
    print("2. Binomial model converges to Black-Scholes with more steps")
    print("3. Monte Carlo is flexible for exotic options and complex payoffs")
    print("4. Greeks measure risk sensitivities crucial for trading")
    print("5. Implied volatility captures market expectations")
    print("6. Performance optimization critical for real-time trading")
