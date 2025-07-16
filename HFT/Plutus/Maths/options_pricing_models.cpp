/*
Plutus HFT Mathematics Problem: Options Pricing Models Implementation
================================================================

Problem Statement:
Implement Black-Scholes option pricing model, Binomial model, and Monte Carlo simulation
for option pricing. Compare results and analyze Greeks.

This is a common quantitative finance problem testing:
1. Mathematical modeling skills
2. Financial derivatives knowledge
3. Numerical methods implementation
4. Performance optimization for HFT

Applications:
- Real-time options pricing in trading systems
- Risk management and hedging strategies
- Market making algorithms
- Volatility surface construction
*/

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>

class OptionsPricingModels {
private:
    // Cumulative standard normal distribution
    double normalCDF(double x) {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }
    
    // Standard normal probability density function
    double normalPDF(double x) {
        return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    }
    
    // Random number generator
    mutable std::mt19937 gen;
    mutable std::normal_distribution<double> normal_dist;

public:
    OptionsPricingModels() : gen(std::chrono::steady_clock::now().time_since_epoch().count()),
                            normal_dist(0.0, 1.0) {}
    
    // Black-Scholes Model
    struct BlackScholesResult {
        double callPrice;
        double putPrice;
        double delta;
        double gamma;
        double theta;
        double vega;
        double rho;
    };
    
    BlackScholesResult blackScholes(double S, double K, double T, double r, double sigma, bool isCall = true) {
        BlackScholesResult result;
        
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        
        double Nd1 = normalCDF(d1);
        double Nd2 = normalCDF(d2);
        double nd1 = normalPDF(d1);
        
        // Call and Put prices
        result.callPrice = S * Nd1 - K * std::exp(-r * T) * Nd2;
        result.putPrice = K * std::exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
        
        // Greeks
        result.delta = isCall ? Nd1 : Nd1 - 1;
        result.gamma = nd1 / (S * sigma * std::sqrt(T));
        result.theta = isCall ? 
            (-S * nd1 * sigma / (2 * std::sqrt(T)) - r * K * std::exp(-r * T) * Nd2) / 365.0 :
            (-S * nd1 * sigma / (2 * std::sqrt(T)) + r * K * std::exp(-r * T) * normalCDF(-d2)) / 365.0;
        result.vega = S * nd1 * std::sqrt(T) / 100.0;
        result.rho = isCall ?
            K * T * std::exp(-r * T) * Nd2 / 100.0 :
            -K * T * std::exp(-r * T) * normalCDF(-d2) / 100.0;
        
        return result;
    }
    
    // Binomial Model
    double binomialOptionPrice(double S, double K, double T, double r, double sigma, 
                              int steps, bool isCall = true) {
        double dt = T / steps;
        double u = std::exp(sigma * std::sqrt(dt));
        double d = 1.0 / u;
        double p = (std::exp(r * dt) - d) / (u - d);
        
        // Initialize option values at maturity
        std::vector<double> optionValues(steps + 1);
        for (int i = 0; i <= steps; i++) {
            double ST = S * std::pow(u, 2 * i - steps);
            optionValues[i] = isCall ? 
                std::max(0.0, ST - K) : 
                std::max(0.0, K - ST);
        }
        
        // Work backwards through the tree
        for (int j = steps - 1; j >= 0; j--) {
            for (int i = 0; i <= j; i++) {
                optionValues[i] = std::exp(-r * dt) * 
                    (p * optionValues[i + 1] + (1 - p) * optionValues[i]);
            }
        }
        
        return optionValues[0];
    }
    
    // Monte Carlo Simulation
    struct MonteCarloResult {
        double price;
        double standardError;
        std::vector<double> paths;
    };
    
    MonteCarloResult monteCarloOptionPrice(double S, double K, double T, double r, double sigma,
                                         int numSimulations, bool isCall = true, bool savePaths = false) {
        MonteCarloResult result;
        std::vector<double> payoffs(numSimulations);
        
        double drift = r - 0.5 * sigma * sigma;
        double diffusion = sigma * std::sqrt(T);
        
        for (int i = 0; i < numSimulations; i++) {
            double z = normal_dist(gen);
            double ST = S * std::exp(drift * T + diffusion * z);
            
            payoffs[i] = isCall ? 
                std::max(0.0, ST - K) : 
                std::max(0.0, K - ST);
            
            if (savePaths && i < 1000) { // Save first 1000 paths for analysis
                result.paths.push_back(ST);
            }
        }
        
        double mean = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / numSimulations;
        result.price = std::exp(-r * T) * mean;
        
        // Calculate standard error
        double variance = 0.0;
        for (double payoff : payoffs) {
            variance += std::pow(payoff - mean, 2);
        }
        variance /= (numSimulations - 1);
        result.standardError = std::sqrt(variance / numSimulations) * std::exp(-r * T);
        
        return result;
    }
    
    // Asian Option Pricing (Monte Carlo)
    double asianOptionPrice(double S, double K, double T, double r, double sigma,
                           int numSimulations, int numSteps, bool isCall = true) {
        double dt = T / numSteps;
        double drift = (r - 0.5 * sigma * sigma) * dt;
        double diffusion = sigma * std::sqrt(dt);
        
        std::vector<double> payoffs(numSimulations);
        
        for (int i = 0; i < numSimulations; i++) {
            double St = S;
            double sum = 0.0;
            
            for (int j = 0; j < numSteps; j++) {
                double z = normal_dist(gen);
                St = St * std::exp(drift + diffusion * z);
                sum += St;
            }
            
            double average = sum / numSteps;
            payoffs[i] = isCall ? 
                std::max(0.0, average - K) : 
                std::max(0.0, K - average);
        }
        
        double mean = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / numSimulations;
        return std::exp(-r * T) * mean;
    }
    
    // Barrier Option Pricing (Monte Carlo)
    double barrierOptionPrice(double S, double K, double T, double r, double sigma,
                             double barrier, int numSimulations, int numSteps,
                             bool isCall = true, bool isKnockOut = true) {
        double dt = T / numSteps;
        double drift = (r - 0.5 * sigma * sigma) * dt;
        double diffusion = sigma * std::sqrt(dt);
        
        std::vector<double> payoffs(numSimulations);
        
        for (int i = 0; i < numSimulations; i++) {
            double St = S;
            bool barrierHit = false;
            
            for (int j = 0; j < numSteps; j++) {
                double z = normal_dist(gen);
                St = St * std::exp(drift + diffusion * z);
                
                if ((isCall && St >= barrier) || (!isCall && St <= barrier)) {
                    barrierHit = true;
                    if (isKnockOut) break;
                }
            }
            
            double intrinsicValue = isCall ? 
                std::max(0.0, St - K) : 
                std::max(0.0, K - St);
            
            if (isKnockOut) {
                payoffs[i] = barrierHit ? 0.0 : intrinsicValue;
            } else {
                payoffs[i] = barrierHit ? intrinsicValue : 0.0;
            }
        }
        
        double mean = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / numSimulations;
        return std::exp(-r * T) * mean;
    }
    
    // Implied Volatility using Newton-Raphson method
    double impliedVolatility(double marketPrice, double S, double K, double T, double r, bool isCall = true) {
        double sigma = 0.2; // Initial guess
        double tolerance = 1e-6;
        int maxIterations = 100;
        
        for (int i = 0; i < maxIterations; i++) {
            BlackScholesResult bs = blackScholes(S, K, T, r, sigma, isCall);
            double price = isCall ? bs.callPrice : bs.putPrice;
            double vega = bs.vega * 100; // Convert from percentage
            
            double priceDiff = price - marketPrice;
            
            if (std::abs(priceDiff) < tolerance) {
                return sigma;
            }
            
            if (std::abs(vega) < 1e-10) {
                break; // Avoid division by zero
            }
            
            sigma = sigma - priceDiff / vega;
            
            // Keep sigma positive
            if (sigma <= 0) {
                sigma = 0.01;
            }
        }
        
        return sigma; // Return best estimate even if not converged
    }
    
    // Portfolio Greeks calculation
    struct PortfolioGreeks {
        double delta;
        double gamma;
        double theta;
        double vega;
        double rho;
    };
    
    PortfolioGreeks calculatePortfolioGreeks(const std::vector<double>& S_values,
                                           const std::vector<double>& K_values,
                                           const std::vector<double>& T_values,
                                           const std::vector<double>& quantities,
                                           double r, double sigma) {
        PortfolioGreeks portfolio = {0, 0, 0, 0, 0};
        
        for (size_t i = 0; i < S_values.size(); i++) {
            BlackScholesResult bs = blackScholes(S_values[i], K_values[i], T_values[i], r, sigma);
            
            portfolio.delta += quantities[i] * bs.delta;
            portfolio.gamma += quantities[i] * bs.gamma;
            portfolio.theta += quantities[i] * bs.theta;
            portfolio.vega += quantities[i] * bs.vega;
            portfolio.rho += quantities[i] * bs.rho;
        }
        
        return portfolio;
    }
};

// Test framework for options pricing models
class OptionsPricingTest {
public:
    static void runTests() {
        std::cout << "Plutus HFT Options Pricing Models Tests" << std::endl;
        std::cout << "=======================================" << std::endl;
        
        testBlackScholes();
        testBinomialModel();
        testMonteCarloSimulation();
        testExoticOptions();
        testImpliedVolatility();
        performanceComparison();
        realWorldScenarios();
    }
    
    static void testBlackScholes() {
        std::cout << "\nBlack-Scholes Model Tests:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        OptionsPricingModels model;
        
        // Standard parameters
        double S = 100.0; // Stock price
        double K = 100.0; // Strike price
        double T = 0.25;  // Time to expiration (3 months)
        double r = 0.05;  // Risk-free rate
        double sigma = 0.2; // Volatility
        
        auto result = model.blackScholes(S, K, T, r, sigma, true);
        
        std::cout << "Call Option Pricing:" << std::endl;
        std::cout << "Stock Price: $" << S << std::endl;
        std::cout << "Strike Price: $" << K << std::endl;
        std::cout << "Time to Expiration: " << T << " years" << std::endl;
        std::cout << "Risk-free Rate: " << (r * 100) << "%" << std::endl;
        std::cout << "Volatility: " << (sigma * 100) << "%" << std::endl;
        std::cout << std::endl;
        
        std::cout << "Results:" << std::endl;
        std::cout << "Call Price: $" << std::fixed << std::setprecision(4) << result.callPrice << std::endl;
        std::cout << "Put Price: $" << result.putPrice << std::endl;
        std::cout << std::endl;
        
        std::cout << "Greeks:" << std::endl;
        std::cout << "Delta: " << result.delta << std::endl;
        std::cout << "Gamma: " << result.gamma << std::endl;
        std::cout << "Theta: " << result.theta << std::endl;
        std::cout << "Vega: " << result.vega << std::endl;
        std::cout << "Rho: " << result.rho << std::endl;
        
        // Put-Call Parity check
        double putCallParity = result.callPrice - result.putPrice - S + K * std::exp(-r * T);
        std::cout << "\nPut-Call Parity Check: " << putCallParity << " (should be ~0)" << std::endl;
        std::cout << "Put-Call Parity: " << (std::abs(putCallParity) < 1e-10 ? "✅" : "❌") << std::endl;
    }
    
    static void testBinomialModel() {
        std::cout << "\nBinomial Model Tests:" << std::endl;
        std::cout << "====================" << std::endl;
        
        OptionsPricingModels model;
        
        double S = 100.0, K = 100.0, T = 0.25, r = 0.05, sigma = 0.2;
        
        std::vector<int> steps = {10, 50, 100, 500, 1000};
        
        for (int n : steps) {
            auto start = std::chrono::high_resolution_clock::now();
            double binomialPrice = model.binomialOptionPrice(S, K, T, r, sigma, n, true);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            std::cout << "Steps: " << n << ", Price: $" << std::fixed << std::setprecision(4) 
                      << binomialPrice << ", Time: " << duration.count() << " μs" << std::endl;
        }
        
        // Compare with Black-Scholes
        auto bs = model.blackScholes(S, K, T, r, sigma, true);
        double binomial1000 = model.binomialOptionPrice(S, K, T, r, sigma, 1000, true);
        double difference = std::abs(bs.callPrice - binomial1000);
        
        std::cout << "\nBlack-Scholes Price: $" << bs.callPrice << std::endl;
        std::cout << "Binomial (1000 steps): $" << binomial1000 << std::endl;
        std::cout << "Difference: $" << difference << std::endl;
        std::cout << "Convergence: " << (difference < 0.01 ? "✅" : "❌") << std::endl;
    }
    
    static void testMonteCarloSimulation() {
        std::cout << "\nMonte Carlo Simulation Tests:" << std::endl;
        std::cout << "============================" << std::endl;
        
        OptionsPricingModels model;
        
        double S = 100.0, K = 100.0, T = 0.25, r = 0.05, sigma = 0.2;
        
        std::vector<int> simulations = {1000, 10000, 100000, 1000000};
        
        for (int n : simulations) {
            auto start = std::chrono::high_resolution_clock::now();
            auto mcResult = model.monteCarloOptionPrice(S, K, T, r, sigma, n, true);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << "Simulations: " << n << ", Price: $" << std::fixed << std::setprecision(4) 
                      << mcResult.price << " ± " << mcResult.standardError 
                      << ", Time: " << duration.count() << " ms" << std::endl;
        }
        
        // Compare with Black-Scholes
        auto bs = model.blackScholes(S, K, T, r, sigma, true);
        auto mc = model.monteCarloOptionPrice(S, K, T, r, sigma, 1000000, true);
        double difference = std::abs(bs.callPrice - mc.price);
        
        std::cout << "\nBlack-Scholes Price: $" << bs.callPrice << std::endl;
        std::cout << "Monte Carlo (1M sims): $" << mc.price << " ± " << mc.standardError << std::endl;
        std::cout << "Difference: $" << difference << std::endl;
        std::cout << "Within 2σ: " << (difference < 2 * mc.standardError ? "✅" : "❌") << std::endl;
    }
    
    static void testExoticOptions() {
        std::cout << "\nExotic Options Tests:" << std::endl;
        std::cout << "====================" << std::endl;
        
        OptionsPricingModels model;
        
        double S = 100.0, K = 100.0, T = 0.25, r = 0.05, sigma = 0.2;
        
        // Asian Option
        double asianPrice = model.asianOptionPrice(S, K, T, r, sigma, 100000, 252, true);
        std::cout << "Asian Call Option Price: $" << std::fixed << std::setprecision(4) << asianPrice << std::endl;
        
        // Barrier Options
        double barrier = 110.0;
        double knockOutPrice = model.barrierOptionPrice(S, K, T, r, sigma, barrier, 100000, 252, true, true);
        double knockInPrice = model.barrierOptionPrice(S, K, T, r, sigma, barrier, 100000, 252, true, false);
        
        std::cout << "Knock-Out Barrier Option (Barrier=$" << barrier << "): $" << knockOutPrice << std::endl;
        std::cout << "Knock-In Barrier Option (Barrier=$" << barrier << "): $" << knockInPrice << std::endl;
        
        // Vanilla option for comparison
        auto vanilla = model.blackScholes(S, K, T, r, sigma, true);
        std::cout << "Vanilla Call Option: $" << vanilla.callPrice << std::endl;
        
        // Check relationship: Knock-Out + Knock-In ≈ Vanilla
        double sum = knockOutPrice + knockInPrice;
        double difference = std::abs(sum - vanilla.callPrice);
        std::cout << "\nKnock-Out + Knock-In = $" << sum << std::endl;
        std::cout << "Relationship check: " << (difference < 0.1 ? "✅" : "❌") << std::endl;
    }
    
    static void testImpliedVolatility() {
        std::cout << "\nImplied Volatility Tests:" << std::endl;
        std::cout << "========================" << std::endl;
        
        OptionsPricingModels model;
        
        double S = 100.0, K = 100.0, T = 0.25, r = 0.05;
        double trueVolatility = 0.25;
        
        // Calculate theoretical price
        auto bs = model.blackScholes(S, K, T, r, trueVolatility, true);
        double marketPrice = bs.callPrice;
        
        // Calculate implied volatility
        double impliedVol = model.impliedVolatility(marketPrice, S, K, T, r, true);
        
        std::cout << "True Volatility: " << (trueVolatility * 100) << "%" << std::endl;
        std::cout << "Market Price: $" << std::fixed << std::setprecision(4) << marketPrice << std::endl;
        std::cout << "Implied Volatility: " << (impliedVol * 100) << "%" << std::endl;
        std::cout << "Difference: " << std::abs(trueVolatility - impliedVol) * 100 << "%" << std::endl;
        std::cout << "Accuracy: " << (std::abs(trueVolatility - impliedVol) < 1e-6 ? "✅" : "❌") << std::endl;
    }
    
    static void performanceComparison() {
        std::cout << "\nPerformance Comparison:" << std::endl;
        std::cout << "======================" << std::endl;
        
        OptionsPricingModels model;
        
        double S = 100.0, K = 100.0, T = 0.25, r = 0.05, sigma = 0.2;
        int iterations = 10000;
        
        // Black-Scholes
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            model.blackScholes(S, K, T, r, sigma, true);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto bsDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Binomial (100 steps)
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            model.binomialOptionPrice(S, K, T, r, sigma, 100, true);
        }
        end = std::chrono::high_resolution_clock::now();
        auto binomialDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Monte Carlo (1000 simulations)
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) { // Fewer iterations due to computational cost
            model.monteCarloOptionPrice(S, K, T, r, sigma, 1000, true);
        }
        end = std::chrono::high_resolution_clock::now();
        auto mcDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Black-Scholes (" << iterations << " calculations): " 
                  << bsDuration.count() << " μs (" << (bsDuration.count() / (double)iterations) << " μs/calc)" << std::endl;
        std::cout << "Binomial 100 steps (" << iterations << " calculations): " 
                  << binomialDuration.count() << " μs (" << (binomialDuration.count() / (double)iterations) << " μs/calc)" << std::endl;
        std::cout << "Monte Carlo 1K sims (100 calculations): " 
                  << mcDuration.count() << " μs (" << (mcDuration.count() / 100.0) << " μs/calc)" << std::endl;
        
        std::cout << "\nSpeed Ranking:" << std::endl;
        std::cout << "1. Black-Scholes (fastest)" << std::endl;
        std::cout << "2. Binomial Model" << std::endl;
        std::cout << "3. Monte Carlo (most flexible)" << std::endl;
    }
    
    static void realWorldScenarios() {
        std::cout << "\nReal-World Trading Scenarios:" << std::endl;
        std::cout << "============================" << std::endl;
        
        OptionsPricingModels model;
        
        // Scenario 1: High volatility environment
        std::cout << "Scenario 1: High Volatility (Market Stress)" << std::endl;
        double S1 = 100.0, K1 = 105.0, T1 = 0.083, r1 = 0.02, sigma1 = 0.8; // 1 month, 80% vol
        auto result1 = model.blackScholes(S1, K1, T1, r1, sigma1, true);
        std::cout << "OTM Call (K=$105): $" << std::fixed << std::setprecision(2) << result1.callPrice << std::endl;
        std::cout << "Delta: " << std::setprecision(3) << result1.delta << ", Vega: " << result1.vega << std::endl;
        
        // Scenario 2: Low volatility environment
        std::cout << "\nScenario 2: Low Volatility (Calm Markets)" << std::endl;
        double sigma2 = 0.1; // 10% vol
        auto result2 = model.blackScholes(S1, K1, T1, r1, sigma2, true);
        std::cout << "OTM Call (K=$105): $" << result2.callPrice << std::endl;
        std::cout << "Delta: " << result2.delta << ", Vega: " << result2.vega << std::endl;
        
        // Scenario 3: Near expiration
        std::cout << "\nScenario 3: Near Expiration (Gamma Risk)" << std::endl;
        double T3 = 0.01; // ~2.5 days
        auto result3 = model.blackScholes(S1, K1, T3, r1, 0.2, true);
        std::cout << "ATM Call (2.5 days): $" << result3.callPrice << std::endl;
        std::cout << "Gamma: " << result3.gamma << ", Theta: " << result3.theta << std::endl;
        
        // Scenario 4: Portfolio Greeks
        std::cout << "\nScenario 4: Portfolio Risk Management" << std::endl;
        std::vector<double> S_vals = {100, 100, 100};
        std::vector<double> K_vals = {95, 100, 105};
        std::vector<double> T_vals = {0.25, 0.25, 0.25};
        std::vector<double> quantities = {100, -200, 100}; // Long 100 puts, short 200 calls, long 100 calls
        
        auto portfolio = model.calculatePortfolioGreeks(S_vals, K_vals, T_vals, quantities, 0.05, 0.2);
        std::cout << "Portfolio Delta: " << std::setprecision(1) << portfolio.delta << std::endl;
        std::cout << "Portfolio Gamma: " << portfolio.gamma << std::endl;
        std::cout << "Portfolio Theta: " << portfolio.theta << " (daily P&L)" << std::endl;
        std::cout << "Portfolio Vega: " << portfolio.vega << std::endl;
        
        bool deltaHedged = std::abs(portfolio.delta) < 10;
        std::cout << "Delta Neutral: " << (deltaHedged ? "✅" : "❌") << std::endl;
    }
};

int main() {
    OptionsPricingTest::runTests();
    return 0;
}

/*
Mathematical Foundations:

Black-Scholes Formula:
C = S₀N(d₁) - Ke^(-rT)N(d₂)
P = Ke^(-rT)N(-d₂) - S₀N(-d₁)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

Greeks:
- Delta (Δ): ∂V/∂S (price sensitivity)
- Gamma (Γ): ∂²V/∂S² (delta sensitivity)
- Theta (Θ): ∂V/∂t (time decay)
- Vega (ν): ∂V/∂σ (volatility sensitivity)
- Rho (ρ): ∂V/∂r (interest rate sensitivity)

Binomial Model:
- Risk-neutral probability: p = (e^(rΔt) - d)/(u - d)
- Up factor: u = e^(σ√Δt)
- Down factor: d = 1/u

Monte Carlo Simulation:
S_T = S₀ × exp((r - σ²/2)T + σ√T × Z)
where Z ~ N(0,1)

Applications in HFT:
1. Real-time option pricing for market making
2. Risk management and position sizing
3. Volatility surface construction
4. Arbitrage opportunity detection
5. Dynamic hedging strategies

Key Insights for Plutus Interview:
1. Understand mathematical foundations
2. Implement efficient numerical methods
3. Consider computational performance
4. Handle edge cases (near expiration, extreme volatility)
5. Real-world trading applications

Common Interview Questions:
1. Explain Black-Scholes assumptions
2. When would you use Monte Carlo vs. Binomial?
3. How do you handle American options?
4. Implement implied volatility calculation
5. Portfolio Greeks and risk management

Optimization Considerations:
- Vectorization for bulk calculations
- Caching of expensive computations
- Parallel Monte Carlo simulations
- GPU acceleration for large portfolios
- Real-time Greeks updates

Risk Management Applications:
- Delta hedging frequency
- Gamma scalping strategies
- Volatility risk management
- Scenario analysis and stress testing
- Portfolio optimization
*/
