"""
JPMorgan HFT Mathematics Problem: Fixed Income Portfolio Risk Analytics (Python)
===============================================================================

Python implementation with NumPy/SciPy for enhanced mathematical computations,
pandas for data handling, and matplotlib for visualization.

Key Features:
- Advanced numerical methods for bond pricing and risk calculations
- Yield curve modeling with multiple interpolation methods
- Monte Carlo simulation for portfolio risk metrics
- Credit risk modeling with transition matrices
- Performance optimization using vectorized operations
- Data visualization and analysis tools

Author: HFT Interview Preparation
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize, interpolate, linalg
from scipy.stats import norm, multivariate_normal
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CreditRating(Enum):
    AAA = 0
    AA = 1
    A = 2
    BBB = 3
    BB = 4
    B = 5
    CCC = 6
    CC = 7
    C = 8
    D = 9

class InterpolationMethod(Enum):
    LINEAR = "linear"
    CUBIC = "cubic"
    NELSON_SIEGEL = "nelson_siegel"
    SVENSSON = "svensson"

class Bond:
    """Enhanced Bond class with comprehensive analytics"""
    
    def __init__(self, bond_id, face_value, coupon_rate, frequency, 
                 time_to_maturity, rating, market_price):
        self.id = bond_id
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.frequency = frequency
        self.time_to_maturity = time_to_maturity
        self.rating = rating
        self.market_price = market_price
        self.yield_to_maturity = self._calculate_ytm()
        
        # Cache analytics for performance
        self._duration = None
        self._convexity = None
        self._dv01 = None
    
    def _calculate_ytm(self, tolerance=1e-8, max_iterations=100):
        """Calculate Yield to Maturity using Brent's method"""
        
        def price_diff(ytm):
            return self.calculate_price_from_yield(ytm) - self.market_price
        
        try:
            # Use scipy's brent method for robust root finding
            ytm = optimize.brentq(price_diff, 0.001, 0.5, xtol=tolerance, maxiter=max_iterations)
            return ytm
        except ValueError:
            # Fallback to Newton-Raphson if Brent fails
            ytm = self.coupon_rate  # Initial guess
            
            for _ in range(max_iterations):
                price = self.calculate_price_from_yield(ytm)
                duration = self.calculate_modified_duration(ytm)
                
                price_diff = price - self.market_price
                if abs(price_diff) < tolerance:
                    break
                
                # Newton-Raphson update
                ytm = ytm - price_diff / (-duration * price)
            
            return ytm
    
    def calculate_price_from_yield(self, ytm):
        """Calculate bond price from yield using vectorized operations"""
        periods = np.arange(1, int(self.time_to_maturity * self.frequency) + 1)
        coupon_payment = (self.coupon_rate * self.face_value) / self.frequency
        
        # Vectorized present value calculation
        discount_factors = (1 + ytm / self.frequency) ** -periods
        coupon_pv = np.sum(coupon_payment * discount_factors)
        
        # Principal present value
        principal_pv = self.face_value * discount_factors[-1]
        
        return coupon_pv + principal_pv
    
    def calculate_modified_duration(self, ytm=None):
        """Calculate modified duration with caching"""
        if self._duration is not None and ytm is None:
            return self._duration
        
        if ytm is None:
            ytm = self.yield_to_maturity
        
        periods = np.arange(1, int(self.time_to_maturity * self.frequency) + 1)
        times = periods / self.frequency
        coupon_payment = (self.coupon_rate * self.face_value) / self.frequency
        
        # Vectorized duration calculation
        discount_factors = (1 + ytm / self.frequency) ** -periods
        cash_flows = np.full_like(periods, coupon_payment, dtype=float)
        cash_flows[-1] += self.face_value  # Add principal to last payment
        
        present_values = cash_flows * discount_factors
        duration = np.sum(times * present_values) / self.market_price
        
        # Modified duration
        modified_duration = duration / (1 + ytm / self.frequency)
        
        if ytm == self.yield_to_maturity:
            self._duration = modified_duration
        
        return modified_duration
    
    def calculate_convexity(self, ytm=None):
        """Calculate convexity with caching"""
        if self._convexity is not None and ytm is None:
            return self._convexity
        
        if ytm is None:
            ytm = self.yield_to_maturity
        
        periods = np.arange(1, int(self.time_to_maturity * self.frequency) + 1)
        times = periods / self.frequency
        coupon_payment = (self.coupon_rate * self.face_value) / self.frequency
        
        # Vectorized convexity calculation
        discount_factors = (1 + ytm / self.frequency) ** -periods
        cash_flows = np.full_like(periods, coupon_payment, dtype=float)
        cash_flows[-1] += self.face_value
        
        present_values = cash_flows * discount_factors
        convexity_terms = times * (times + 1.0 / self.frequency) * present_values
        
        convexity = np.sum(convexity_terms) / (self.market_price * (1 + ytm / self.frequency) ** 2)
        
        if ytm == self.yield_to_maturity:
            self._convexity = convexity
        
        return convexity
    
    def calculate_dv01(self):
        """Calculate DV01 (Dollar Value of 01 basis point)"""
        if self._dv01 is not None:
            return self._dv01
        
        duration = self.calculate_modified_duration()
        dv01 = duration * self.market_price * 0.0001  # 1 basis point = 0.01%
        
        self._dv01 = dv01
        return dv01
    
    def price_sensitivity_analysis(self, yield_shocks=None):
        """Analyze price sensitivity to yield changes"""
        if yield_shocks is None:
            yield_shocks = np.arange(-0.02, 0.021, 0.005)  # -200bp to +200bp
        
        base_ytm = self.yield_to_maturity
        duration = self.calculate_modified_duration()
        convexity = self.calculate_convexity()
        
        results = []
        for shock in yield_shocks:
            new_ytm = base_ytm + shock
            actual_price = self.calculate_price_from_yield(new_ytm)
            
            # Duration approximation
            duration_price = self.market_price * (1 - duration * shock)
            
            # Duration + Convexity approximation
            convexity_price = self.market_price * (1 - duration * shock + 0.5 * convexity * shock**2)
            
            results.append({
                'yield_shock_bp': shock * 10000,
                'new_yield': new_ytm * 100,
                'actual_price': actual_price,
                'duration_approx': duration_price,
                'convexity_approx': convexity_price,
                'duration_error': abs(actual_price - duration_price),
                'convexity_error': abs(actual_price - convexity_price)
            })
        
        return pd.DataFrame(results)

class YieldCurve:
    """Advanced yield curve with multiple interpolation methods"""
    
    def __init__(self, maturities, yields, method=InterpolationMethod.CUBIC):
        self.maturities = np.array(maturities)
        self.yields = np.array(yields)
        self.method = method
        
        # Sort by maturity
        sort_idx = np.argsort(self.maturities)
        self.maturities = self.maturities[sort_idx]
        self.yields = self.yields[sort_idx]
        
        # Fit interpolation function
        self._fit_curve()
    
    def _fit_curve(self):
        """Fit the yield curve based on the specified method"""
        if self.method == InterpolationMethod.LINEAR:
            self.interpolator = interpolate.interp1d(self.maturities, self.yields, 
                                                   kind='linear', 
                                                   bounds_error=False, 
                                                   fill_value='extrapolate')
        
        elif self.method == InterpolationMethod.CUBIC:
            self.interpolator = interpolate.CubicSpline(self.maturities, self.yields,
                                                      bc_type='natural')
        
        elif self.method == InterpolationMethod.NELSON_SIEGEL:
            self._fit_nelson_siegel()
        
        elif self.method == InterpolationMethod.SVENSSON:
            self._fit_svensson()
    
    def _fit_nelson_siegel(self):
        """Fit Nelson-Siegel model: y(t) = β₀ + β₁*f₁(t,τ) + β₂*f₂(t,τ)"""
        
        def nelson_siegel(t, beta0, beta1, beta2, tau):
            """Nelson-Siegel functional form"""
            term1 = (1 - np.exp(-t / tau)) / (t / tau)
            term2 = term1 - np.exp(-t / tau)
            return beta0 + beta1 * term1 + beta2 * term2
        
        # Fit parameters using least squares
        try:
            popt, _ = optimize.curve_fit(nelson_siegel, self.maturities, self.yields,
                                       p0=[0.05, -0.02, 0.01, 2.0],
                                       bounds=([-0.1, -0.1, -0.1, 0.1], 
                                              [0.2, 0.1, 0.1, 10.0]))
            self.ns_params = popt
            self.interpolator = lambda t: nelson_siegel(t, *popt)
        except:
            # Fallback to cubic spline
            self.method = InterpolationMethod.CUBIC
            self._fit_curve()
    
    def _fit_svensson(self):
        """Fit Svensson model (extended Nelson-Siegel)"""
        
        def svensson(t, beta0, beta1, beta2, beta3, tau1, tau2):
            """Svensson functional form"""
            term1 = (1 - np.exp(-t / tau1)) / (t / tau1)
            term2 = term1 - np.exp(-t / tau1)
            term3 = (1 - np.exp(-t / tau2)) / (t / tau2) - np.exp(-t / tau2)
            return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3
        
        # Fit parameters using least squares
        try:
            popt, _ = optimize.curve_fit(svensson, self.maturities, self.yields,
                                       p0=[0.05, -0.02, 0.01, 0.005, 2.0, 5.0],
                                       bounds=([-0.1, -0.1, -0.1, -0.1, 0.1, 0.1], 
                                              [0.2, 0.1, 0.1, 0.1, 10.0, 20.0]))
            self.svensson_params = popt
            self.interpolator = lambda t: svensson(t, *popt)
        except:
            # Fallback to Nelson-Siegel
            self.method = InterpolationMethod.NELSON_SIEGEL
            self._fit_curve()
    
    def get_yield(self, maturity):
        """Get yield for given maturity"""
        if np.isscalar(maturity):
            return float(self.interpolator(maturity))
        else:
            return self.interpolator(maturity)
    
    def get_forward_rate(self, t1, t2):
        """Calculate forward rate between t1 and t2"""
        y1 = self.get_yield(t1)
        y2 = self.get_yield(t2)
        
        forward_rate = ((1 + y2) ** t2 / (1 + y1) ** t1) ** (1 / (t2 - t1)) - 1
        return forward_rate
    
    def plot_curve(self, title="Yield Curve", save_path=None):
        """Plot the yield curve"""
        t_fine = np.linspace(self.maturities.min(), self.maturities.max(), 100)
        y_fine = self.get_yield(t_fine)
        
        plt.figure(figsize=(12, 8))
        plt.plot(t_fine, y_fine * 100, 'b-', label=f'Fitted ({self.method.value})', linewidth=2)
        plt.scatter(self.maturities, self.yields * 100, color='red', s=50, 
                   label='Market Data', zorder=5)
        
        plt.xlabel('Time to Maturity (Years)', fontsize=12)
        plt.ylabel('Yield (%)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class PrincipalComponentAnalysis:
    """PCA for yield curve analysis"""
    
    def __init__(self):
        self.n_components = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance_ratio = None
        self.mean_yields = None
    
    def fit(self, yield_changes):
        """Fit PCA model to yield changes matrix"""
        yield_changes = np.array(yield_changes)
        
        # Center the data
        self.mean_yields = np.mean(yield_changes, axis=0)
        centered_data = yield_changes - self.mean_yields
        
        # Calculate covariance matrix
        cov_matrix = np.cov(centered_data.T)
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]
        
        # Calculate explained variance ratio
        total_variance = np.sum(self.eigenvalues)
        self.explained_variance_ratio = self.eigenvalues / total_variance
        
        self.n_components = len(self.eigenvalues)
    
    def transform(self, yield_changes, n_components=None):
        """Transform yield changes to PC space"""
        if n_components is None:
            n_components = self.n_components
        
        yield_changes = np.array(yield_changes)
        centered_data = yield_changes - self.mean_yields
        
        return centered_data @ self.eigenvectors[:, :n_components]
    
    def get_loadings(self, n_components=3):
        """Get factor loadings for first n components"""
        return self.eigenvectors[:, :n_components]
    
    def plot_components(self, maturities, n_components=3, save_path=None):
        """Plot the first n principal components"""
        loadings = self.get_loadings(n_components)
        
        plt.figure(figsize=(12, 8))
        
        component_names = ['Level', 'Slope', 'Curvature']
        colors = ['blue', 'red', 'green']
        
        for i in range(n_components):
            variance_pct = self.explained_variance_ratio[i] * 100
            label = f'PC{i+1} ({component_names[i]}) - {variance_pct:.1f}%'
            plt.plot(maturities, loadings[:, i], 'o-', 
                    color=colors[i], label=label, linewidth=2, markersize=6)
        
        plt.xlabel('Time to Maturity (Years)', fontsize=12)
        plt.ylabel('Factor Loading', fontsize=12)
        plt.title('Principal Components of Yield Curve', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class CreditTransitionMatrix:
    """Credit transition matrix for migration risk"""
    
    def __init__(self):
        self.ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
        
        # Typical 1-year transition matrix (from S&P/Moody's data)
        self.transition_matrix = np.array([
            # To:  AAA    AA     A    BBB    BB     B    CCC    CC     C     D
            [0.950, 0.040, 0.008, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # AAA
            [0.010, 0.920, 0.060, 0.008, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000],  # AA
            [0.000, 0.020, 0.900, 0.070, 0.008, 0.002, 0.000, 0.000, 0.000, 0.000],  # A
            [0.000, 0.000, 0.030, 0.870, 0.080, 0.018, 0.002, 0.000, 0.000, 0.000],  # BBB
            [0.000, 0.000, 0.000, 0.050, 0.800, 0.120, 0.025, 0.003, 0.002, 0.000],  # BB
            [0.000, 0.000, 0.000, 0.000, 0.080, 0.750, 0.120, 0.030, 0.015, 0.005],  # B
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.150, 0.600, 0.150, 0.080, 0.020],  # CCC
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.200, 0.500, 0.200, 0.100],  # CC
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.300, 0.400, 0.300],  # C
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]   # D
        ])
    
    def get_transition_probability(self, from_rating, to_rating, time_horizon=1):
        """Get transition probability from one rating to another"""
        from_idx = self.ratings.index(from_rating)
        to_idx = self.ratings.index(to_rating)
        
        if time_horizon == 1:
            return self.transition_matrix[from_idx, to_idx]
        else:
            # Matrix exponentiation for multi-period
            multi_period_matrix = np.linalg.matrix_power(self.transition_matrix, time_horizon)
            return multi_period_matrix[from_idx, to_idx]
    
    def get_default_probabilities(self, time_horizons=[1, 3, 5, 10]):
        """Calculate cumulative default probabilities"""
        default_probs = {}
        
        for horizon in time_horizons:
            probs = []
            multi_period_matrix = np.linalg.matrix_power(self.transition_matrix, horizon)
            
            for i in range(len(self.ratings) - 1):  # Exclude D rating
                default_prob = multi_period_matrix[i, -1]  # Probability to default
                probs.append(default_prob)
            
            default_probs[f'{horizon}Y'] = probs
        
        return pd.DataFrame(default_probs, index=self.ratings[:-1])
    
    def plot_transition_heatmap(self, save_path=None):
        """Plot transition matrix as heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Create mask for upper triangle (ratings can't improve by more than a few notches typically)
        mask = np.triu(np.ones_like(self.transition_matrix, dtype=bool), k=3)
        
        sns.heatmap(self.transition_matrix, 
                   xticklabels=self.ratings,
                   yticklabels=self.ratings,
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Transition Probability'},
                   mask=mask * (self.transition_matrix < 0.001))
        
        plt.title('Credit Rating Transition Matrix (1 Year)', fontsize=14, fontweight='bold')
        plt.xlabel('To Rating', fontsize=12)
        plt.ylabel('From Rating', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class FixedIncomePortfolioAnalytics:
    """Main portfolio analytics engine"""
    
    def __init__(self):
        self.pca = PrincipalComponentAnalysis()
        self.credit_matrix = CreditTransitionMatrix()
    
    def calculate_portfolio_metrics(self, bonds, weights, yield_curve=None):
        """Calculate comprehensive portfolio risk metrics"""
        
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Portfolio-level metrics
        portfolio_duration = 0.0
        portfolio_convexity = 0.0
        portfolio_dv01 = 0.0
        total_value = 0.0
        
        for bond, weight in zip(bonds, weights):
            bond_value = bond.market_price * weight
            total_value += bond_value
            
            duration = bond.calculate_modified_duration()
            convexity = bond.calculate_convexity()
            dv01 = bond.calculate_dv01()
            
            portfolio_duration += (bond_value * duration)
            portfolio_convexity += (bond_value * convexity)
            portfolio_dv01 += (weight * dv01)
        
        portfolio_duration /= total_value
        portfolio_convexity /= total_value
        
        # Risk metrics via Monte Carlo
        var_metrics = self._calculate_var_monte_carlo(bonds, weights, yield_curve)
        
        return {
            'portfolio_duration': portfolio_duration,
            'portfolio_convexity': portfolio_convexity,
            'portfolio_dv01': portfolio_dv01,
            'total_value': total_value,
            **var_metrics
        }
    
    def _calculate_var_monte_carlo(self, bonds, weights, yield_curve, n_simulations=10000):
        """Calculate VaR using Monte Carlo simulation"""
        
        # Generate correlated yield shocks
        n_bonds = len(bonds)
        
        # Simplified correlation structure (in practice, use historical data)
        correlation_matrix = np.eye(n_bonds)
        for i in range(n_bonds):
            for j in range(i+1, n_bonds):
                # Correlation decreases with maturity difference
                maturity_diff = abs(bonds[i].time_to_maturity - bonds[j].time_to_maturity)
                correlation = max(0.3, 0.9 - 0.1 * maturity_diff)
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        # Generate random shocks
        np.random.seed(42)  # For reproducibility
        
        # Daily yield volatilities (simplified)
        volatilities = np.array([0.008 + 0.002 * np.exp(-bond.time_to_maturity / 5) 
                                for bond in bonds])
        
        # Generate correlated shocks
        mvn = multivariate_normal(mean=np.zeros(n_bonds), cov=correlation_matrix)
        random_shocks = mvn.rvs(n_simulations)
        yield_shocks = random_shocks * volatilities
        
        # Calculate portfolio returns for each scenario
        portfolio_returns = np.zeros(n_simulations)
        
        for i, shocks in enumerate(yield_shocks):
            portfolio_return = 0.0
            
            for j, (bond, weight, shock) in enumerate(zip(bonds, weights, shocks)):
                # Calculate price change using duration and convexity
                duration = bond.calculate_modified_duration()
                convexity = bond.calculate_convexity()
                
                price_change = -duration * shock + 0.5 * convexity * shock**2
                portfolio_return += weight * price_change
            
            portfolio_returns[i] = portfolio_return
        
        # Calculate VaR and Expected Shortfall
        portfolio_returns_sorted = np.sort(portfolio_returns)
        
        var_95 = -np.percentile(portfolio_returns_sorted, 5)
        var_99 = -np.percentile(portfolio_returns_sorted, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95_idx = int(0.05 * n_simulations)
        es_99_idx = int(0.01 * n_simulations)
        
        es_95 = -np.mean(portfolio_returns_sorted[:es_95_idx])
        es_99 = -np.mean(portfolio_returns_sorted[:es_99_idx])
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'portfolio_returns': portfolio_returns
        }
    
    def perform_pca_analysis(self, yield_changes_df):
        """Perform PCA on historical yield changes"""
        self.pca.fit(yield_changes_df.values)
        return {
            'explained_variance_ratio': self.pca.explained_variance_ratio,
            'loadings': self.pca.get_loadings(),
            'eigenvalues': self.pca.eigenvalues
        }
    
    def plot_portfolio_risk_analysis(self, metrics, save_path=None):
        """Create comprehensive risk analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Return distribution
        returns = metrics['portfolio_returns']
        axes[0, 0].hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(-metrics['var_95'], color='red', linestyle='--', 
                          label=f"95% VaR: {metrics['var_95']:.3f}")
        axes[0, 0].axvline(-metrics['var_99'], color='darkred', linestyle='--',
                          label=f"99% VaR: {metrics['var_99']:.3f}")
        axes[0, 0].set_xlabel('Portfolio Return')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Portfolio Return Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Risk metrics comparison
        risk_metrics = ['VaR 95%', 'VaR 99%', 'ES 95%', 'ES 99%']
        risk_values = [metrics['var_95'], metrics['var_99'], 
                      metrics['expected_shortfall_95'], metrics['expected_shortfall_99']]
        
        bars = axes[0, 1].bar(risk_metrics, risk_values, color=['blue', 'navy', 'red', 'darkred'])
        axes[0, 1].set_ylabel('Risk Metric Value')
        axes[0, 1].set_title('Portfolio Risk Metrics')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, risk_values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Cumulative return distribution
        sorted_returns = np.sort(returns)
        percentiles = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns) * 100
        
        axes[1, 0].plot(sorted_returns, percentiles, 'b-', linewidth=2)
        axes[1, 0].axvline(-metrics['var_95'], color='red', linestyle='--', alpha=0.7)
        axes[1, 0].axvline(-metrics['var_99'], color='darkred', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(5, color='red', linestyle=':', alpha=0.7)
        axes[1, 0].axhline(1, color='darkred', linestyle=':', alpha=0.7)
        axes[1, 0].set_xlabel('Portfolio Return')
        axes[1, 0].set_ylabel('Cumulative Probability (%)')
        axes[1, 0].set_title('Cumulative Distribution Function')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Risk decomposition
        duration = metrics['portfolio_duration']
        convexity = metrics['portfolio_convexity']
        dv01 = metrics['portfolio_dv01']
        
        risk_components = ['Duration Risk', 'Convexity Effect', 'DV01']
        risk_component_values = [duration * 100, convexity * 10000, dv01 * 100]
        
        axes[1, 1].pie(np.abs(risk_component_values), labels=risk_components, autopct='%1.1f%%',
                      colors=['lightcoral', 'lightblue', 'lightgreen'])
        axes[1, 1].set_title('Risk Component Breakdown')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Test and demonstration functions
def create_sample_portfolio():
    """Create a sample bond portfolio for testing"""
    
    bonds = [
        Bond("TREASURY_2Y", 1000, 0.020, 2, 2.0, CreditRating.AAA, 995),
        Bond("TREASURY_5Y", 1000, 0.025, 2, 5.0, CreditRating.AAA, 985),
        Bond("TREASURY_10Y", 1000, 0.030, 2, 10.0, CreditRating.AAA, 970),
        Bond("CORPORATE_3Y", 1000, 0.040, 2, 3.0, CreditRating.A, 1010),
        Bond("CORPORATE_7Y", 1000, 0.045, 2, 7.0, CreditRating.BBB, 1020),
        Bond("CORPORATE_15Y", 1000, 0.050, 2, 15.0, CreditRating.BBB, 1030)
    ]
    
    weights = [0.25, 0.20, 0.15, 0.20, 0.15, 0.05]
    
    return bonds, weights

def create_sample_yield_curve():
    """Create a sample yield curve"""
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    yields = [0.015, 0.018, 0.022, 0.025, 0.027, 0.028, 0.029, 0.030, 0.031, 0.032]
    
    return YieldCurve(maturities, yields, InterpolationMethod.CUBIC)

def generate_sample_yield_changes():
    """Generate sample historical yield changes for PCA"""
    np.random.seed(42)
    
    # Simulate 252 days of yield changes for 10 maturities
    n_days = 252
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    
    # Create realistic correlation structure
    correlation_matrix = np.eye(len(maturities))
    for i in range(len(maturities)):
        for j in range(i+1, len(maturities)):
            correlation = 0.9 * np.exp(-abs(i - j) * 0.2)
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation
    
    # Generate correlated yield changes
    mvn = multivariate_normal(mean=np.zeros(len(maturities)), cov=correlation_matrix)
    yield_changes = mvn.rvs(n_days) * 0.01  # Scale to realistic volatility
    
    return pd.DataFrame(yield_changes, columns=[f'{m}Y' for m in maturities])

def run_comprehensive_test():
    """Run comprehensive test of all functionality"""
    
    print("JPMorgan Fixed Income Portfolio Analytics - Python Implementation")
    print("=" * 70)
    
    # 1. Create sample portfolio
    print("\n1. Creating Sample Portfolio...")
    bonds, weights = create_sample_portfolio()
    
    # Display portfolio details
    portfolio_df = pd.DataFrame([
        {
            'Bond ID': bond.id,
            'Face Value': bond.face_value,
            'Coupon Rate': f"{bond.coupon_rate:.3f}",
            'Maturity': f"{bond.time_to_maturity:.1f}Y",
            'Rating': bond.rating.name,
            'Market Price': f"${bond.market_price:.2f}",
            'YTM': f"{bond.yield_to_maturity:.3f}",
            'Duration': f"{bond.calculate_modified_duration():.2f}",
            'Convexity': f"{bond.calculate_convexity():.1f}",
            'Weight': f"{weight:.1%}"
        }
        for bond, weight in zip(bonds, weights)
    ])
    
    print(portfolio_df.to_string(index=False))
    
    # 2. Bond analytics
    print("\n2. Individual Bond Analysis...")
    sample_bond = bonds[2]  # 10Y Treasury
    sensitivity_df = sample_bond.price_sensitivity_analysis()
    print(f"\nPrice Sensitivity Analysis for {sample_bond.id}:")
    print(sensitivity_df.round(4))
    
    # 3. Yield curve analysis
    print("\n3. Yield Curve Analysis...")
    yield_curve = create_sample_yield_curve()
    
    # Test different interpolation methods
    test_maturities = [1.5, 4.0, 8.5, 25.0]
    print("\nYield Curve Interpolation Comparison:")
    print("Maturity | Linear | Cubic | Nelson-Siegel")
    print("-" * 40)
    
    for method in [InterpolationMethod.LINEAR, InterpolationMethod.CUBIC, InterpolationMethod.NELSON_SIEGEL]:
        curve = YieldCurve(yield_curve.maturities, yield_curve.yields, method)
        method_yields = [curve.get_yield(m) for m in test_maturities]
        
        if method == InterpolationMethod.LINEAR:
            linear_yields = method_yields
        elif method == InterpolationMethod.CUBIC:
            cubic_yields = method_yields
        else:
            ns_yields = method_yields
    
    for i, maturity in enumerate(test_maturities):
        print(f"{maturity:7.1f}Y | {linear_yields[i]:5.3f} | {cubic_yields[i]:5.3f} | {ns_yields[i]:5.3f}")
    
    # 4. Portfolio risk metrics
    print("\n4. Portfolio Risk Analysis...")
    analytics = FixedIncomePortfolioAnalytics()
    metrics = analytics.calculate_portfolio_metrics(bonds, weights, yield_curve)
    
    print(f"\nPortfolio Metrics:")
    print(f"Portfolio Duration: {metrics['portfolio_duration']:.3f} years")
    print(f"Portfolio Convexity: {metrics['portfolio_convexity']:.2f}")
    print(f"Portfolio DV01: ${metrics['portfolio_dv01']:.2f}")
    print(f"Total Portfolio Value: ${metrics['total_value']:.2f}")
    print(f"\nRisk Metrics:")
    print(f"95% VaR: {metrics['var_95']:.3f} ({metrics['var_95']*100:.1f}%)")
    print(f"99% VaR: {metrics['var_99']:.3f} ({metrics['var_99']*100:.1f}%)")
    print(f"95% Expected Shortfall: {metrics['expected_shortfall_95']:.3f}")
    print(f"99% Expected Shortfall: {metrics['expected_shortfall_99']:.3f}")
    
    # 5. PCA Analysis
    print("\n5. Principal Component Analysis...")
    yield_changes_df = generate_sample_yield_changes()
    pca_results = analytics.perform_pca_analysis(yield_changes_df)
    
    print("Explained Variance Ratio:")
    for i, ratio in enumerate(pca_results['explained_variance_ratio'][:5]):
        component_names = ['Level', 'Slope', 'Curvature', 'Butterfly', 'Twist']
        print(f"PC{i+1} ({component_names[i]}): {ratio:.1%}")
    
    # 6. Credit Risk Analysis
    print("\n6. Credit Risk Analysis...")
    credit_matrix = CreditTransitionMatrix()
    default_probs_df = credit_matrix.get_default_probabilities()
    
    print("Cumulative Default Probabilities:")
    print(default_probs_df.round(4))
    
    # Create visualizations
    print("\n7. Creating Visualizations...")
    
    # Plot yield curve
    yield_curve.plot_curve("Sample Yield Curve - Multiple Interpolation Methods")
    
    # Plot PCA components
    maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
    analytics.pca.plot_components(maturities)
    
    # Plot credit transition matrix
    credit_matrix.plot_transition_heatmap()
    
    # Plot portfolio risk analysis
    analytics.plot_portfolio_risk_analysis(metrics)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("\nKey Features Demonstrated:")
    print("✓ Advanced bond pricing and analytics")
    print("✓ Multiple yield curve interpolation methods")
    print("✓ Monte Carlo VaR calculation")
    print("✓ Principal Component Analysis")
    print("✓ Credit risk modeling")
    print("✓ Comprehensive visualization")
    print("✓ Performance optimization with NumPy")

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    import warnings
    warnings.filterwarnings('ignore')
    
    # Run the comprehensive test
    run_comprehensive_test()
