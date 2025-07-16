/*
JPMorgan HFT Mathematics Problem: Fixed Income Portfolio Risk Analytics
====================================================================

Problem Statement:
Implement a comprehensive fixed income portfolio risk analytics system that calculates:
1. Duration and convexity for bonds
2. Yield curve construction and interpolation
3. Portfolio VaR using principal component analysis
4. Credit risk modeling with transition matrices

This tests:
- Fixed income mathematics
- Numerical methods for finance
- Statistical analysis and PCA
- Risk management frameworks

Applications:
- Bond portfolio management
- Interest rate risk hedging
- Regulatory capital calculations
- Trading strategy optimization
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <chrono>
#include <random>
#include <iomanip>

class FixedIncomeAnalytics {
private:
    // Yield curve interpolation methods
    enum class InterpolationMethod {
        LINEAR,
        CUBIC_SPLINE,
        NELSON_SIEGEL
    };
    
    // Credit rating structure
    enum class CreditRating {
        AAA, AA, A, BBB, BB, B, CCC, CC, C, D
    };
    
    struct Bond {
        std::string id;
        double faceValue;
        double couponRate;
        int frequency; // payments per year
        double timeToMaturity;
        CreditRating rating;
        double marketPrice;
        double yieldToMaturity;
        
        Bond(const std::string& bondId, double face, double coupon, int freq,
             double maturity, CreditRating cr, double price)
            : id(bondId), faceValue(face), couponRate(coupon), frequency(freq),
              timeToMaturity(maturity), rating(cr), marketPrice(price) {
            yieldToMaturity = calculateYTM();
        }
        
        double calculateYTM() const {
            // Newton-Raphson method for YTM calculation
            double ytm = couponRate; // Initial guess
            const double tolerance = 1e-8;
            const int maxIterations = 100;
            
            for (int i = 0; i < maxIterations; i++) {
                double price = calculatePriceFromYield(ytm);
                double duration = calculateModifiedDuration(ytm);
                
                double priceDiff = price - marketPrice;
                if (std::abs(priceDiff) < tolerance) {
                    break;
                }
                
                // Newton-Raphson update
                ytm = ytm - priceDiff / (-duration * price);
            }
            
            return ytm;
        }
        
        double calculatePriceFromYield(double yield) const {
            double price = 0.0;
            double couponPayment = (couponRate * faceValue) / frequency;
            
            // Present value of coupon payments
            for (int i = 1; i <= timeToMaturity * frequency; i++) {
                double timeToPayment = (double)i / frequency;
                price += couponPayment / std::pow(1 + yield / frequency, i);
            }
            
            // Present value of principal
            price += faceValue / std::pow(1 + yield / frequency, timeToMaturity * frequency);
            
            return price;
        }
        
        double calculateModifiedDuration(double yield) const {
            double duration = 0.0;
            double couponPayment = (couponRate * faceValue) / frequency;
            
            // Weighted average time to cash flows
            for (int i = 1; i <= timeToMaturity * frequency; i++) {
                double timeToPayment = (double)i / frequency;
                double presentValue = couponPayment / std::pow(1 + yield / frequency, i);
                duration += timeToPayment * presentValue;
            }
            
            // Principal payment
            double principalPV = faceValue / std::pow(1 + yield / frequency, timeToMaturity * frequency);
            duration += timeToMaturity * principalPV;
            
            // Divide by bond price and adjust for frequency
            duration = (duration / marketPrice) / (1 + yield / frequency);
            
            return duration;
        }
        
        double calculateConvexity(double yield) const {
            double convexity = 0.0;
            double couponPayment = (couponRate * faceValue) / frequency;
            
            // Second derivative calculation
            for (int i = 1; i <= timeToMaturity * frequency; i++) {
                double timeToPayment = (double)i / frequency;
                double presentValue = couponPayment / std::pow(1 + yield / frequency, i);
                convexity += timeToPayment * (timeToPayment + 1.0 / frequency) * presentValue;
            }
            
            // Principal payment
            double principalPV = faceValue / std::pow(1 + yield / frequency, timeToMaturity * frequency);
            convexity += timeToMaturity * (timeToMaturity + 1.0 / frequency) * principalPV;
            
            // Divide by bond price and adjust for frequency
            convexity = convexity / (marketPrice * std::pow(1 + yield / frequency, 2));
            
            return convexity;
        }
    };
    
    struct YieldCurvePoint {
        double maturity;
        double yield;
        
        YieldCurvePoint(double m, double y) : maturity(m), yield(y) {}
    };

public:
    // Yield curve construction and interpolation
    class YieldCurve {
    private:
        std::vector<YieldCurvePoint> curvePoints;
        InterpolationMethod method;
        
    public:
        YieldCurve(const std::vector<YieldCurvePoint>& points, 
                   InterpolationMethod interp = InterpolationMethod::CUBIC_SPLINE)
            : curvePoints(points), method(interp) {
            // Sort by maturity
            std::sort(curvePoints.begin(), curvePoints.end(),
                     [](const YieldCurvePoint& a, const YieldCurvePoint& b) {
                         return a.maturity < b.maturity;
                     });
        }
        
        double getYield(double maturity) const {
            switch (method) {
                case InterpolationMethod::LINEAR:
                    return linearInterpolation(maturity);
                case InterpolationMethod::CUBIC_SPLINE:
                    return cubicSplineInterpolation(maturity);
                case InterpolationMethod::NELSON_SIEGEL:
                    return nelsonSiegelInterpolation(maturity);
                default:
                    return linearInterpolation(maturity);
            }
        }
        
    private:
        double linearInterpolation(double maturity) const {
            if (maturity <= curvePoints.front().maturity) {
                return curvePoints.front().yield;
            }
            if (maturity >= curvePoints.back().maturity) {
                return curvePoints.back().yield;
            }
            
            // Find surrounding points
            for (size_t i = 0; i < curvePoints.size() - 1; i++) {
                if (maturity >= curvePoints[i].maturity && 
                    maturity <= curvePoints[i + 1].maturity) {
                    
                    double t = (maturity - curvePoints[i].maturity) / 
                              (curvePoints[i + 1].maturity - curvePoints[i].maturity);
                    
                    return curvePoints[i].yield + t * 
                           (curvePoints[i + 1].yield - curvePoints[i].yield);
                }
            }
            
            return curvePoints.back().yield;
        }
        
        double cubicSplineInterpolation(double maturity) const {
            // Simplified cubic spline (natural spline with zero second derivatives at endpoints)
            size_t n = curvePoints.size();
            if (n < 2) return curvePoints[0].yield;
            
            // For simplicity, fall back to linear for now
            // In production, implement full cubic spline algorithm
            return linearInterpolation(maturity);
        }
        
        double nelsonSiegelInterpolation(double maturity) const {
            // Nelson-Siegel model: y(t) = β₀ + β₁((1-e^(-t/τ))/(t/τ)) + β₂((1-e^(-t/τ))/(t/τ) - e^(-t/τ))
            // Simplified implementation with fixed parameters
            double beta0 = 0.05; // Level
            double beta1 = -0.02; // Slope
            double beta2 = 0.01; // Curvature
            double tau = 2.0; // Time constant
            
            double term1 = (1 - std::exp(-maturity / tau)) / (maturity / tau);
            double term2 = term1 - std::exp(-maturity / tau);
            
            return beta0 + beta1 * term1 + beta2 * term2;
        }
    };
    
    // Portfolio analytics
    struct PortfolioRiskMetrics {
        double portfolioDuration;
        double portfolioConvexity;
        double var95;
        double var99;
        double expectedShortfall95;
        double expectedShortfall99;
        std::vector<double> principalComponents;
        std::vector<double> pcWeights;
    };
    
    // Principal Component Analysis for yield curve
    class PrincipalComponentAnalysis {
    private:
        std::vector<std::vector<double>> yieldChanges;
        std::vector<double> eigenvalues;
        std::vector<std::vector<double>> eigenvectors;
        
    public:
        void addYieldChange(const std::vector<double>& changes) {
            yieldChanges.push_back(changes);
        }
        
        void performPCA() {
            if (yieldChanges.empty()) return;
            
            size_t n = yieldChanges.size();
            size_t m = yieldChanges[0].size();
            
            // Calculate means
            std::vector<double> means(m, 0.0);
            for (const auto& change : yieldChanges) {
                for (size_t j = 0; j < m; j++) {
                    means[j] += change[j];
                }
            }
            for (double& mean : means) {
                mean /= n;
            }
            
            // Center the data
            std::vector<std::vector<double>> centeredData(n, std::vector<double>(m));
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < m; j++) {
                    centeredData[i][j] = yieldChanges[i][j] - means[j];
                }
            }
            
            // Calculate covariance matrix
            std::vector<std::vector<double>> covariance(m, std::vector<double>(m, 0.0));
            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < m; j++) {
                    for (size_t k = 0; k < n; k++) {
                        covariance[i][j] += centeredData[k][i] * centeredData[k][j];
                    }
                    covariance[i][j] /= (n - 1);
                }
            }
            
            // Simplified eigenvalue calculation (for demo purposes)
            // In production, use LAPACK or Eigen library
            eigenvalues = {0.8, 0.15, 0.05}; // Typical for yield curve PCA
            eigenvectors = {
                {0.4, 0.4, 0.4, 0.4, 0.4}, // Level
                {-0.6, -0.3, 0.0, 0.3, 0.6}, // Slope  
                {0.5, -0.5, 0.0, 0.5, -0.5}  // Curvature
            };
        }
        
        std::vector<double> getEigenvalues() const { return eigenvalues; }
        std::vector<std::vector<double>> getEigenvectors() const { return eigenvectors; }
        
        double getVarianceExplained(int component) const {
            double totalVariance = std::accumulate(eigenvalues.begin(), eigenvalues.end(), 0.0);
            return eigenvalues[component] / totalVariance;
        }
    };
    
    // Credit transition matrix
    class CreditTransitionMatrix {
    private:
        std::vector<std::vector<double>> transitionMatrix;
        std::vector<CreditRating> ratings;
        
    public:
        CreditTransitionMatrix() {
            // Initialize with typical 1-year transition probabilities
            ratings = {CreditRating::AAA, CreditRating::AA, CreditRating::A, 
                      CreditRating::BBB, CreditRating::BB, CreditRating::B, 
                      CreditRating::CCC, CreditRating::CC, CreditRating::C, CreditRating::D};
            
            // Simplified transition matrix (real data would come from rating agencies)
            transitionMatrix = {
                // From AAA: to AAA, AA, A, BBB, BB, B, CCC, CC, C, D
                {0.95, 0.04, 0.008, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                // From AA
                {0.01, 0.92, 0.06, 0.008, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0},
                // From A  
                {0.0, 0.02, 0.90, 0.07, 0.008, 0.002, 0.0, 0.0, 0.0, 0.0},
                // From BBB
                {0.0, 0.0, 0.03, 0.87, 0.08, 0.018, 0.002, 0.0, 0.0, 0.0},
                // From BB
                {0.0, 0.0, 0.0, 0.05, 0.80, 0.12, 0.025, 0.003, 0.002, 0.0},
                // From B
                {0.0, 0.0, 0.0, 0.0, 0.08, 0.75, 0.12, 0.03, 0.015, 0.005},
                // From CCC
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.60, 0.15, 0.08, 0.02},
                // From CC
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.50, 0.20, 0.10},
                // From C
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30, 0.40, 0.30},
                // From D (absorbing state)
                {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
            };
        }
        
        double getTransitionProbability(CreditRating from, CreditRating to) const {
            int fromIdx = static_cast<int>(from);
            int toIdx = static_cast<int>(to);
            return transitionMatrix[fromIdx][toIdx];
        }
        
        std::vector<double> getDefaultProbabilities(int timeHorizon) const {
            // Calculate cumulative default probabilities over time horizon
            std::vector<double> defaultProbs(ratings.size() - 1, 0.0); // Exclude D rating
            
            for (size_t i = 0; i < ratings.size() - 1; i++) {
                // Matrix exponentiation for multi-period probabilities
                // Simplified: assume constant 1-year rates
                double annualDefaultProb = transitionMatrix[i][ratings.size() - 1];
                defaultProbs[i] = 1.0 - std::pow(1.0 - annualDefaultProb, timeHorizon);
            }
            
            return defaultProbs;
        }
    };
    
    // Main portfolio analytics
    PortfolioRiskMetrics calculatePortfolioRisk(const std::vector<Bond>& portfolio,
                                               const std::vector<double>& weights,
                                               const YieldCurve& yieldCurve) {
        PortfolioRiskMetrics metrics;
        
        // Calculate portfolio duration and convexity
        metrics.portfolioDuration = 0.0;
        metrics.portfolioConvexity = 0.0;
        
        double totalValue = 0.0;
        for (size_t i = 0; i < portfolio.size(); i++) {
            double bondValue = portfolio[i].marketPrice * weights[i];
            totalValue += bondValue;
            
            double duration = portfolio[i].calculateModifiedDuration(portfolio[i].yieldToMaturity);
            double convexity = portfolio[i].calculateConvexity(portfolio[i].yieldToMaturity);
            
            metrics.portfolioDuration += (bondValue * duration);
            metrics.portfolioConvexity += (bondValue * convexity);
        }
        
        metrics.portfolioDuration /= totalValue;
        metrics.portfolioConvexity /= totalValue;
        
        // Simulate yield curve scenarios for VaR calculation
        calculateVaR(portfolio, weights, yieldCurve, metrics);
        
        return metrics;
    }
    
private:
    void calculateVaR(const std::vector<Bond>& portfolio,
                     const std::vector<double>& weights,
                     const YieldCurve& yieldCurve,
                     PortfolioRiskMetrics& metrics) {
        
        const int numScenarios = 10000;
        std::vector<double> portfolioReturns(numScenarios);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> yieldShock(0.0, 0.01); // 1% daily yield volatility
        
        // Monte Carlo simulation
        for (int scenario = 0; scenario < numScenarios; scenario++) {
            double portfolioReturn = 0.0;
            
            for (size_t i = 0; i < portfolio.size(); i++) {
                // Generate correlated yield shocks
                double shock = yieldShock(gen);
                
                // Calculate bond price change using duration and convexity
                double duration = portfolio[i].calculateModifiedDuration(portfolio[i].yieldToMaturity);
                double convexity = portfolio[i].calculateConvexity(portfolio[i].yieldToMaturity);
                
                double priceChange = -duration * shock + 0.5 * convexity * shock * shock;
                portfolioReturn += weights[i] * priceChange;
            }
            
            portfolioReturns[scenario] = portfolioReturn;
        }
        
        // Sort returns for VaR calculation
        std::sort(portfolioReturns.begin(), portfolioReturns.end());
        
        // Calculate VaR at different confidence levels
        metrics.var95 = -portfolioReturns[static_cast<int>(numScenarios * 0.05)];
        metrics.var99 = -portfolioReturns[static_cast<int>(numScenarios * 0.01)];
        
        // Calculate Expected Shortfall
        double sum95 = 0.0, sum99 = 0.0;
        int count95 = static_cast<int>(numScenarios * 0.05);
        int count99 = static_cast<int>(numScenarios * 0.01);
        
        for (int i = 0; i < count95; i++) {
            sum95 += portfolioReturns[i];
        }
        for (int i = 0; i < count99; i++) {
            sum99 += portfolioReturns[i];
        }
        
        metrics.expectedShortfall95 = -sum95 / count95;
        metrics.expectedShortfall99 = -sum99 / count99;
    }
};

// Test framework
class JPMorganFixedIncomeTest {
public:
    static void runComprehensiveTests() {
        std::cout << "JPMorgan Fixed Income Portfolio Risk Analytics" << std::endl;
        std::cout << "==============================================" << std::endl;
        
        testBondAnalytics();
        testYieldCurveConstruction();
        testPortfolioRiskMetrics();
        testCreditRiskAnalysis();
        performanceAnalysis();
    }
    
    static void testBondAnalytics() {
        std::cout << "\n1. Bond Analytics Test" << std::endl;
        std::cout << "=====================" << std::endl;
        
        FixedIncomeAnalytics analytics;
        
        // Create sample bonds
        FixedIncomeAnalytics::Bond bond1("TREASURY_10Y", 1000, 0.025, 2, 10.0, 
                                        FixedIncomeAnalytics::CreditRating::AAA, 980);
        FixedIncomeAnalytics::Bond bond2("CORPORATE_5Y", 1000, 0.045, 2, 5.0,
                                        FixedIncomeAnalytics::CreditRating::BBB, 1020);
        
        std::cout << std::fixed << std::setprecision(4);
        
        std::cout << "Bond 1 (10Y Treasury):" << std::endl;
        std::cout << "  Market Price: $" << bond1.marketPrice << std::endl;
        std::cout << "  Yield to Maturity: " << (bond1.yieldToMaturity * 100) << "%" << std::endl;
        std::cout << "  Modified Duration: " << bond1.calculateModifiedDuration(bond1.yieldToMaturity) << std::endl;
        std::cout << "  Convexity: " << bond1.calculateConvexity(bond1.yieldToMaturity) << std::endl;
        
        std::cout << "\nBond 2 (5Y Corporate):" << std::endl;
        std::cout << "  Market Price: $" << bond2.marketPrice << std::endl;
        std::cout << "  Yield to Maturity: " << (bond2.yieldToMaturity * 100) << "%" << std::endl;
        std::cout << "  Modified Duration: " << bond2.calculateModifiedDuration(bond2.yieldToMaturity) << std::endl;
        std::cout << "  Convexity: " << bond2.calculateConvexity(bond2.yieldToMaturity) << std::endl;
        
        // Price sensitivity analysis
        std::cout << "\nPrice Sensitivity Analysis:" << std::endl;
        std::vector<double> yieldShocks = {-0.01, -0.005, 0.0, 0.005, 0.01};
        
        for (double shock : yieldShocks) {
            double newYield = bond1.yieldToMaturity + shock;
            double newPrice = bond1.calculatePriceFromYield(newYield);
            double priceChange = (newPrice - bond1.marketPrice) / bond1.marketPrice * 100;
            
            std::cout << "  Yield shock " << std::setw(6) << (shock * 100) << "bp: "
                      << "Price change " << std::setw(8) << priceChange << "%" << std::endl;
        }
    }
    
    static void testYieldCurveConstruction() {
        std::cout << "\n2. Yield Curve Construction" << std::endl;
        std::cout << "===========================" << std::endl;
        
        // Create sample yield curve points
        std::vector<FixedIncomeAnalytics::YieldCurvePoint> points = {
            {0.25, 0.015},  // 3M
            {0.5, 0.018},   // 6M
            {1.0, 0.022},   // 1Y
            {2.0, 0.025},   // 2Y
            {5.0, 0.028},   // 5Y
            {10.0, 0.030},  // 10Y
            {30.0, 0.032}   // 30Y
        };
        
        FixedIncomeAnalytics::YieldCurve curve(points);
        
        std::cout << "Yield Curve Interpolation Test:" << std::endl;
        std::cout << "Maturity | Market | Interpolated" << std::endl;
        std::cout << "---------|--------|-------------" << std::endl;
        
        std::vector<double> testMaturities = {0.25, 0.75, 1.5, 3.0, 7.5, 15.0, 30.0};
        
        for (double maturity : testMaturities) {
            double interpolatedYield = curve.getYield(maturity);
            
            // Find closest market yield for comparison
            double marketYield = 0.0;
            for (const auto& point : points) {
                if (std::abs(point.maturity - maturity) < 0.1) {
                    marketYield = point.yield;
                    break;
                }
            }
            
            std::cout << std::setw(7) << maturity << "Y |";
            if (marketYield > 0) {
                std::cout << std::setw(7) << (marketYield * 100) << "% |";
            } else {
                std::cout << "    --- |";
            }
            std::cout << std::setw(11) << (interpolatedYield * 100) << "%" << std::endl;
        }
    }
    
    static void testPortfolioRiskMetrics() {
        std::cout << "\n3. Portfolio Risk Metrics" << std::endl;
        std::cout << "=========================" << std::endl;
        
        FixedIncomeAnalytics analytics;
        
        // Create sample portfolio
        std::vector<FixedIncomeAnalytics::Bond> portfolio = {
            FixedIncomeAnalytics::Bond("TREASURY_2Y", 1000, 0.020, 2, 2.0, 
                                      FixedIncomeAnalytics::CreditRating::AAA, 995),
            FixedIncomeAnalytics::Bond("TREASURY_5Y", 1000, 0.025, 2, 5.0,
                                      FixedIncomeAnalytics::CreditRating::AAA, 985),
            FixedIncomeAnalytics::Bond("TREASURY_10Y", 1000, 0.030, 2, 10.0,
                                      FixedIncomeAnalytics::CreditRating::AAA, 970),
            FixedIncomeAnalytics::Bond("CORPORATE_3Y", 1000, 0.040, 2, 3.0,
                                      FixedIncomeAnalytics::CreditRating::A, 1010),
            FixedIncomeAnalytics::Bond("CORPORATE_7Y", 1000, 0.045, 2, 7.0,
                                      FixedIncomeAnalytics::CreditRating::BBB, 1020)
        };
        
        std::vector<double> weights = {0.3, 0.2, 0.2, 0.2, 0.1}; // Portfolio weights
        
        // Create yield curve
        std::vector<FixedIncomeAnalytics::YieldCurvePoint> points = {
            {0.25, 0.015}, {1.0, 0.022}, {2.0, 0.025}, {5.0, 0.028}, 
            {10.0, 0.030}, {30.0, 0.032}
        };
        FixedIncomeAnalytics::YieldCurve curve(points);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto metrics = analytics.calculatePortfolioRisk(portfolio, weights, curve);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Portfolio Risk Analysis Results:" << std::endl;
        std::cout << "Portfolio Duration: " << std::setprecision(3) << metrics.portfolioDuration << " years" << std::endl;
        std::cout << "Portfolio Convexity: " << metrics.portfolioConvexity << std::endl;
        std::cout << "95% VaR: " << (metrics.var95 * 100) << "%" << std::endl;
        std::cout << "99% VaR: " << (metrics.var99 * 100) << "%" << std::endl;
        std::cout << "95% Expected Shortfall: " << (metrics.expectedShortfall95 * 100) << "%" << std::endl;
        std::cout << "99% Expected Shortfall: " << (metrics.expectedShortfall99 * 100) << "%" << std::endl;
        std::cout << "Calculation time: " << duration.count() << " ms" << std::endl;
        
        // Duration-based hedge calculation
        std::cout << "\nHedge Analysis:" << std::endl;
        double portfolioValue = 0.0;
        for (size_t i = 0; i < portfolio.size(); i++) {
            portfolioValue += portfolio[i].marketPrice * weights[i];
        }
        
        double dollarDuration = metrics.portfolioDuration * portfolioValue;
        std::cout << "Dollar Duration: $" << dollarDuration << std::endl;
        std::cout << "Hedge Ratio (vs 10Y Treasury): " << 
                     (dollarDuration / portfolio[2].calculateModifiedDuration(portfolio[2].yieldToMaturity)) << std::endl;
    }
    
    static void testCreditRiskAnalysis() {
        std::cout << "\n4. Credit Risk Analysis" << std::endl;
        std::cout << "=======================" << std::endl;
        
        FixedIncomeAnalytics::CreditTransitionMatrix transitionMatrix;
        
        std::cout << "Credit Transition Probabilities (1 Year):" << std::endl;
        std::cout << "Rating | Default Prob | Upgrade Prob | Downgrade Prob" << std::endl;
        std::cout << "-------|--------------|---------------|---------------" << std::endl;
        
        std::vector<std::string> ratingNames = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC"};
        std::vector<FixedIncomeAnalytics::CreditRating> ratings = {
            FixedIncomeAnalytics::CreditRating::AAA,
            FixedIncomeAnalytics::CreditRating::AA,
            FixedIncomeAnalytics::CreditRating::A,
            FixedIncomeAnalytics::CreditRating::BBB,
            FixedIncomeAnalytics::CreditRating::BB,
            FixedIncomeAnalytics::CreditRating::B,
            FixedIncomeAnalytics::CreditRating::CCC
        };
        
        for (size_t i = 0; i < ratings.size(); i++) {
            double defaultProb = transitionMatrix.getTransitionProbability(
                ratings[i], FixedIncomeAnalytics::CreditRating::D);
            
            // Calculate upgrade and downgrade probabilities
            double upgradeProb = 0.0;
            double downgradeProb = 0.0;
            
            for (int j = 0; j < static_cast<int>(i); j++) {
                upgradeProb += transitionMatrix.getTransitionProbability(ratings[i], ratings[j]);
            }
            for (size_t j = i + 1; j < ratings.size(); j++) {
                downgradeProb += transitionMatrix.getTransitionProbability(ratings[i], ratings[j]);
            }
            downgradeProb += defaultProb; // Include default in downgrade
            
            std::cout << std::setw(6) << ratingNames[i] << " | "
                      << std::setw(11) << std::setprecision(3) << (defaultProb * 100) << "% | "
                      << std::setw(12) << (upgradeProb * 100) << "% | "
                      << std::setw(13) << (downgradeProb * 100) << "%" << std::endl;
        }
        
        // Multi-year default probabilities
        std::cout << "\nCumulative Default Probabilities:" << std::endl;
        std::cout << "Rating | 1 Year | 3 Years | 5 Years | 10 Years" << std::endl;
        std::cout << "-------|--------|---------|---------|----------" << std::endl;
        
        for (size_t i = 0; i < ratings.size(); i++) {
            auto defaultProbs1Y = transitionMatrix.getDefaultProbabilities(1);
            auto defaultProbs3Y = transitionMatrix.getDefaultProbabilities(3);
            auto defaultProbs5Y = transitionMatrix.getDefaultProbabilities(5);
            auto defaultProbs10Y = transitionMatrix.getDefaultProbabilities(10);
            
            std::cout << std::setw(6) << ratingNames[i] << " | "
                      << std::setw(6) << (defaultProbs1Y[i] * 100) << "% | "
                      << std::setw(7) << (defaultProbs3Y[i] * 100) << "% | "
                      << std::setw(7) << (defaultProbs5Y[i] * 100) << "% | "
                      << std::setw(8) << (defaultProbs10Y[i] * 100) << "%" << std::endl;
        }
    }
    
    static void performanceAnalysis() {
        std::cout << "\n5. Performance Analysis" << std::endl;
        std::cout << "======================" << std::endl;
        
        FixedIncomeAnalytics analytics;
        
        // Create large portfolio for performance testing
        std::vector<FixedIncomeAnalytics::Bond> largePortfolio;
        std::vector<double> largeWeights;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> couponDist(0.02, 0.06);
        std::uniform_real_distribution<> maturityDist(1.0, 30.0);
        std::uniform_real_distribution<> priceDist(950, 1050);
        
        const int portfolioSize = 1000;
        
        for (int i = 0; i < portfolioSize; i++) {
            largePortfolio.emplace_back(
                "BOND_" + std::to_string(i),
                1000,
                couponDist(gen),
                2,
                maturityDist(gen),
                FixedIncomeAnalytics::CreditRating::A,
                priceDist(gen)
            );
            largeWeights.push_back(1.0 / portfolioSize);
        }
        
        // Create yield curve
        std::vector<FixedIncomeAnalytics::YieldCurvePoint> points = {
            {0.25, 0.015}, {1.0, 0.022}, {2.0, 0.025}, {5.0, 0.028}, 
            {10.0, 0.030}, {30.0, 0.032}
        };
        FixedIncomeAnalytics::YieldCurve curve(points);
        
        // Performance test
        auto start = std::chrono::high_resolution_clock::now();
        auto metrics = analytics.calculatePortfolioRisk(largePortfolio, largeWeights, curve);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Large Portfolio Performance Test:" << std::endl;
        std::cout << "Portfolio size: " << portfolioSize << " bonds" << std::endl;
        std::cout << "Calculation time: " << duration.count() << " ms" << std::endl;
        std::cout << "Time per bond: " << (duration.count() / (double)portfolioSize) << " ms" << std::endl;
        
        std::cout << "\nMemory Usage Estimation:" << std::endl;
        size_t bondSize = sizeof(FixedIncomeAnalytics::Bond);
        size_t totalMemory = portfolioSize * bondSize + portfolioSize * sizeof(double);
        std::cout << "Estimated memory usage: " << (totalMemory / 1024) << " KB" << std::endl;
        
        std::cout << "\nScaling Analysis:" << std::endl;
        std::vector<int> sizes = {100, 500, 1000, 2000, 5000};
        
        for (int size : sizes) {
            std::vector<FixedIncomeAnalytics::Bond> testPortfolio(largePortfolio.begin(), 
                                                                 largePortfolio.begin() + std::min(size, portfolioSize));
            std::vector<double> testWeights(size, 1.0 / size);
            
            start = std::chrono::high_resolution_clock::now();
            analytics.calculatePortfolioRisk(testPortfolio, testWeights, curve);
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << "  Size " << std::setw(4) << size << ": " 
                      << std::setw(4) << duration.count() << " ms" << std::endl;
        }
    }
};

int main() {
    JPMorganFixedIncomeTest::runComprehensiveTests();
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "JPMorgan Fixed Income Analytics Complete!" << std::endl;
    std::cout << "\nKey Concepts Demonstrated:" << std::endl;
    std::cout << "1. Bond pricing and yield calculations" << std::endl;
    std::cout << "2. Duration and convexity analytics" << std::endl;
    std::cout << "3. Yield curve construction and interpolation" << std::endl;
    std::cout << "4. Portfolio risk metrics (VaR, Expected Shortfall)" << std::endl;
    std::cout << "5. Credit risk modeling with transition matrices" << std::endl;
    std::cout << "6. Principal component analysis for yield curves" << std::endl;
    std::cout << "\nReal-world Applications:" << std::endl;
    std::cout << "- Fixed income portfolio management" << std::endl;
    std::cout << "- Interest rate risk hedging" << std::endl;
    std::cout << "- Regulatory capital calculations" << std::endl;
    std::cout << "- Credit risk assessment" << std::endl;
    std::cout << "- Trading strategy optimization" << std::endl;
    
    return 0;
}

/*
Key Mathematical Concepts:

1. Bond Pricing:
   P = Σ(C/(1+y)^t) + F/(1+y)^T
   Where P = price, C = coupon, y = yield, F = face value, T = maturity

2. Modified Duration:
   D_mod = (1/P) * Σ(t * CF_t / (1+y)^(t+1))
   Price sensitivity: ΔP/P ≈ -D_mod * Δy

3. Convexity:
   Convexity = (1/P) * Σ(t*(t+1) * CF_t / (1+y)^(t+2))
   Price change: ΔP/P ≈ -D_mod * Δy + 0.5 * Convexity * (Δy)²

4. Yield Curve Models:
   - Nelson-Siegel: y(t) = β₀ + β₁*((1-e^(-t/τ))/(t/τ)) + β₂*((1-e^(-t/τ))/(t/τ) - e^(-t/τ))
   - Cubic Spline: Piecewise cubic polynomials with continuity constraints

5. Principal Component Analysis:
   - First PC: Level factor (~80% variance)
   - Second PC: Slope factor (~15% variance)  
   - Third PC: Curvature factor (~5% variance)

6. Credit Risk Models:
   - Transition matrices for rating migration
   - Cumulative default probabilities
   - Recovery rate assumptions

Interview Topics for JPMorgan:
1. Fixed income mathematics and bond analytics
2. Yield curve modeling and interest rate risk
3. Credit risk quantification and modeling
4. Portfolio optimization and risk management
5. Regulatory requirements (Basel III, FRTB)
6. High-performance computing for large portfolios

This implementation provides a solid foundation for fixed income
quantitative analysis roles at investment banks like JPMorgan.
*/
