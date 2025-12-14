from Portfolio_Optimizer import ImpliedVolatilityPortfolioOptimizer
from portfolio_monte_carlo import MonteCarloSimulator

def run_full_analysis(
    tickers,
    # Optimizer settings
    lookback_years=3,
    risk_free_rate=0.04,
    iv_multiplier=0.7,
    blend_alpha=0.6,
    use_black_litterman=True,
    bl_tau=0.05,
    min_weight=0.0,
    max_weight=0.35,
    optimize_method="max_sharpe",
    use_intrinsic_value=True,
    # Monte Carlo settings
    initial_capital=500000,
    simulation_years=10,
    num_simulations=10000,
    rebalance_frequency='W',
):

    print("RUNNING PORTFOLIO OPTIMIZER")

    # Create and run optimizer
    optimizer = ImpliedVolatilityPortfolioOptimizer(
        tickers=tickers,
        lookback_years=lookback_years,
        risk_free_rate=risk_free_rate,
        iv_multiplier=iv_multiplier,
        blend_alpha=blend_alpha,
        use_black_litterman=use_black_litterman,
        bl_tau=bl_tau,
        min_weight=min_weight,
        max_weight=max_weight,
        optimize_method=optimize_method,
        use_intrinsic_value=use_intrinsic_value,
    )

    optimizer.run_optimization()
    optimizer.display_results()

    print("RUNNING MONTE CARLO SIMULATION")

    # Create Monte Carlo simulator with optimizer outputs
    simulator = MonteCarloSimulator(
        expected_returns=optimizer.mu_final,
        covariance_matrix=optimizer.cov_matrix,
        weights=optimizer.weights,
        initial_capital=initial_capital,
        years=simulation_years,
        num_simulations=num_simulations,
        rebalance_frequency=rebalance_frequency,
        risk_free_rate=risk_free_rate,
    )

    # Run simulation
    simulator.run_simulation()
    metrics = simulator.print_results()

    # Run scenario analysis
    print("SCENARIO ANALYSIS")
    scenarios = simulator.run_scenario_analysis()
    simulator.print_scenario_results(scenarios)

    # Export results
    optimizer.export_results("optimizer_results.json")
    simulator.export_results("monte_carlo_results.json")

    return optimizer, simulator, metrics


if __name__ == "__main__":
    # Your tickers
    tickers = [
        "EXC", "UNH", "META", "FCX", "VZ", "QQQ", "IWM", "VTI",
        "VXUS", "TLT", "IEF", "SHY", "AGG", "LQD", "ARKK", "IPO", "HIMS"
    ]

    # Run the full analysis
    optimizer, simulator, metrics = run_full_analysis(
        tickers=tickers,
        # Optimizer settings
        lookback_years=3,
        risk_free_rate=0.04,
        iv_multiplier=0.7,
        blend_alpha=0.6,
        use_black_litterman=True,
        bl_tau=0.05,
        min_weight=0.0,
        max_weight=1.0,
        optimize_method="max_sharpe",
        use_intrinsic_value=True,
        # Monte Carlo settings
        initial_capital=500000,
        simulation_years=10,
        num_simulations=10000,
        rebalance_frequency='W',  # Weekly rebalancing
    )

    print("ANALYSIS COMPLETE")