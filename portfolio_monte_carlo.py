import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")


class MonteCarloSimulator:
    def __init__(
            self,
            expected_returns,
            covariance_matrix,
            weights,
            initial_capital=500000,
            years=10,
            num_simulations=10000,
            rebalance_frequency='W',
            risk_free_rate=0.04,
            trading_days_per_year=252,
    ):
        self.tickers = list(expected_returns.index)
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.weights = weights
        self.initial_capital = initial_capital
        self.years = years
        self.num_simulations = num_simulations
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.num_periods = int(years * trading_days_per_year)

        self.simulation_paths = None
        self.terminal_values = None
        self.drawdowns = None
        self.metrics = None

    def get_rebalance_days(self):
        if self.rebalance_frequency == 'D':
            return list(range(self.num_periods))
        elif self.rebalance_frequency == 'W':
            return list(range(0, self.num_periods, 5))
        elif self.rebalance_frequency == 'M':
            return list(range(0, self.num_periods, 21))
        elif self.rebalance_frequency == 'Q':
            return list(range(0, self.num_periods, 63))
        elif self.rebalance_frequency == 'Y':
            return list(range(0, self.num_periods, 252))
        else:
            return list(range(0, self.num_periods, 5))

    def run_simulation(self):
        print(f"\nRunning Monte Carlo Simulation")
        print(f"  Simulations: {self.num_simulations:,}")
        print(f"  Time horizon: {self.years} years ({self.num_periods} trading days)")
        print(f"  Initial capital: ${self.initial_capital:,}")
        print(f"  Rebalance frequency: {self.rebalance_frequency}")

        n_assets = len(self.tickers)

        # Convert annual to daily
        daily_returns = self.expected_returns.values / self.trading_days_per_year
        daily_cov = self.covariance_matrix.values / self.trading_days_per_year

        # Cholesky decomposition for correlated returns
        try:
            cholesky = np.linalg.cholesky(daily_cov)
        except np.linalg.LinAlgError:
            daily_cov += np.eye(n_assets) * 1e-8
            cholesky = np.linalg.cholesky(daily_cov)

        w = self.weights.values
        all_paths = np.zeros((self.num_simulations, self.num_periods + 1))
        all_paths[:, 0] = self.initial_capital
        rebalance_days = set(self.get_rebalance_days())

        print("  Simulating paths")

        for sim in range(self.num_simulations):
            portfolio_value = self.initial_capital
            asset_values = portfolio_value * w

            for day in range(self.num_periods):
                z = np.random.standard_normal(n_assets)
                correlated_z = cholesky @ z
                asset_daily_returns = daily_returns + correlated_z
                asset_values = asset_values * (1 + asset_daily_returns)
                portfolio_value = asset_values.sum()

                if day in rebalance_days:
                    asset_values = portfolio_value * w

                all_paths[sim, day + 1] = portfolio_value

            if (sim + 1) % 2000 == 0:
                print(f"    Completed {sim + 1:,} / {self.num_simulations:,} simulations")

        self.simulation_paths = all_paths
        self.terminal_values = all_paths[:, -1]
        self.calculate_drawdowns()

        print("  Simulation complete!")
        return self.simulation_paths

    def calculate_drawdowns(self):
        drawdowns = np.zeros(self.num_simulations)
        for sim in range(self.num_simulations):
            path = self.simulation_paths[sim]
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            drawdowns[sim] = drawdown.min()
        self.drawdowns = drawdowns

    def calculate_metrics(self):
        if self.simulation_paths is None:
            raise ValueError("Run simulation first")

        terminal = self.terminal_values
        initial = self.initial_capital

        total_returns = (terminal / initial) - 1
        annualized_returns = (terminal / initial) ** (1 / self.years) - 1

        portfolio_expected_return = (self.weights @ self.expected_returns)
        portfolio_variance = self.weights @ self.covariance_matrix @ self.weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        portfolio_sharpe = (portfolio_expected_return - self.risk_free_rate) / portfolio_volatility

        metrics = {
            'expected_annual_return': portfolio_expected_return,
            'expected_annual_volatility': portfolio_volatility,
            'expected_sharpe_ratio': portfolio_sharpe,
            'terminal_mean': terminal.mean(),
            'terminal_median': np.median(terminal),
            'terminal_std': terminal.std(),
            'terminal_min': terminal.min(),
            'terminal_max': terminal.max(),
            'terminal_5th_percentile': np.percentile(terminal, 5),
            'terminal_25th_percentile': np.percentile(terminal, 25),
            'terminal_75th_percentile': np.percentile(terminal, 75),
            'terminal_95th_percentile': np.percentile(terminal, 95),
            'total_return_mean': total_returns.mean(),
            'total_return_median': np.median(total_returns),
            'annualized_return_mean': annualized_returns.mean(),
            'annualized_return_median': np.median(annualized_returns),
            'annualized_return_5th': np.percentile(annualized_returns, 5),
            'annualized_return_95th': np.percentile(annualized_returns, 95),
            'probability_of_loss': (terminal < initial).mean(),
            'probability_of_doubling': (terminal >= 2 * initial).mean(),
            'probability_of_50pct_gain': (terminal >= 1.5 * initial).mean(),
            'VaR_5pct': initial - np.percentile(terminal, 5),
            'VaR_1pct': initial - np.percentile(terminal, 1),
            'CVaR_5pct': initial - terminal[terminal <= np.percentile(terminal, 5)].mean(),
            'max_drawdown_mean': self.drawdowns.mean(),
            'max_drawdown_median': np.median(self.drawdowns),
            'max_drawdown_5th_percentile': np.percentile(self.drawdowns, 5),
            'max_drawdown_worst': self.drawdowns.min(),
            'probability_of_20pct_drawdown': (self.drawdowns <= -0.20).mean(),
            'probability_of_30pct_drawdown': (self.drawdowns <= -0.30).mean(),
            'probability_of_50pct_drawdown': (self.drawdowns <= -0.50).mean(),
        }

        self.metrics = metrics
        return metrics

    def run_scenario_analysis(self):
        print("\nRunning Scenario Analysis")

        scenarios = {
            'bull_market': self.run_scenario_sim(1.5, 0.8, "Bull Market"),
            'bear_market': self.run_scenario_sim(0.3, 1.5, "Bear Market"),
            'high_volatility': self.run_scenario_sim(1.0, 2.0, "High Volatility"),
            'stagflation': self.run_scenario_sim(0.2, 1.8, "Stagflation"),
        }

        return scenarios

    def run_scenario_sim(self, return_multiplier, vol_multiplier, name, num_sims=1000):
        print(f"  {name}: {return_multiplier}x returns, {vol_multiplier}x volatility")

        n_assets = len(self.tickers)
        daily_returns = (self.expected_returns.values * return_multiplier) / self.trading_days_per_year
        daily_cov = (self.covariance_matrix.values * vol_multiplier ** 2) / self.trading_days_per_year

        try:
            cholesky = np.linalg.cholesky(daily_cov)
        except np.linalg.LinAlgError:
            daily_cov += np.eye(n_assets) * 1e-8
            cholesky = np.linalg.cholesky(daily_cov)

        w = self.weights.values
        rebalance_days = set(self.get_rebalance_days())

        terminal_values = np.zeros(num_sims)
        max_drawdowns = np.zeros(num_sims)

        for sim in range(num_sims):
            path = np.zeros(self.num_periods + 1)
            path[0] = self.initial_capital
            asset_values = self.initial_capital * w

            for day in range(self.num_periods):
                z = np.random.standard_normal(n_assets)
                correlated_z = cholesky @ z
                asset_daily_returns = daily_returns + correlated_z
                asset_values = asset_values * (1 + asset_daily_returns)
                portfolio_value = asset_values.sum()

                if day in rebalance_days:
                    asset_values = portfolio_value * w

                path[day + 1] = portfolio_value

            terminal_values[sim] = path[-1]
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_drawdowns[sim] = drawdown.min()

        return {
            'terminal_median': np.median(terminal_values),
            'terminal_5th': np.percentile(terminal_values, 5),
            'terminal_95th': np.percentile(terminal_values, 95),
            'annualized_return_median': (np.median(terminal_values) / self.initial_capital) ** (1 / self.years) - 1,
            'probability_of_loss': (terminal_values < self.initial_capital).mean(),
            'max_drawdown_median': np.median(max_drawdowns),
            'max_drawdown_5th': np.percentile(max_drawdowns, 5),
        }

    def print_results(self):
        if self.metrics is None:
            self.calculate_metrics()

        m = self.metrics

        print("MONTE CARLO SIMULATION RESULTS")

        print(f"\nSimulation Parameters:")
        print(f"  Initial Capital:     ${self.initial_capital:,.0f}")
        print(f"  Time Horizon:        {self.years} years")
        print(f"  Simulations:         {self.num_simulations:,}")
        print(f"  Rebalance Freq:      {self.rebalance_frequency}")

        print(f"\nPortfolio Assumptions (from your optimizer):")
        print(f"  Expected Return:     {m['expected_annual_return']:.1%} annually")
        print(f"  Expected Volatility: {m['expected_annual_volatility']:.1%} annually")
        print(f"  Expected Sharpe:     {m['expected_sharpe_ratio']:.2f}")

        print("TERMINAL VALUE DISTRIBUTION (after {:.0f} years)".format(self.years))
        print(
            f"  Worst Case (5th %ile):   ${m['terminal_5th_percentile']:>12,.0f}  ({(m['terminal_5th_percentile'] / self.initial_capital - 1):+.1%})")
        print(
            f"  25th Percentile:         ${m['terminal_25th_percentile']:>12,.0f}  ({(m['terminal_25th_percentile'] / self.initial_capital - 1):+.1%})")
        print(
            f"  Median:                  ${m['terminal_median']:>12,.0f}  ({(m['terminal_median'] / self.initial_capital - 1):+.1%})")
        print(
            f"  Mean:                    ${m['terminal_mean']:>12,.0f}  ({(m['terminal_mean'] / self.initial_capital - 1):+.1%})")
        print(
            f"  75th Percentile:         ${m['terminal_75th_percentile']:>12,.0f}  ({(m['terminal_75th_percentile'] / self.initial_capital - 1):+.1%})")
        print(
            f"  Best Case (95th %ile):   ${m['terminal_95th_percentile']:>12,.0f}  ({(m['terminal_95th_percentile'] / self.initial_capital - 1):+.1%})")

        print("ANNUALIZED RETURN DISTRIBUTION")
        print(f"  5th Percentile:      {m['annualized_return_5th']:>8.1%}")
        print(f"  Median:              {m['annualized_return_median']:>8.1%}")
        print(f"  Mean:                {m['annualized_return_mean']:>8.1%}")
        print(f"  95th Percentile:     {m['annualized_return_95th']:>8.1%}")

        print("PROBABILITY ANALYSIS")
        print(f"  Probability of Loss:          {m['probability_of_loss']:>8.1%}")
        print(f"  Probability of 50%+ Gain:     {m['probability_of_50pct_gain']:>8.1%}")
        print(f"  Probability of Doubling:      {m['probability_of_doubling']:>8.1%}")

        print("RISK METRICS")
        print(f"  Value at Risk (5%):           ${m['VaR_5pct']:>12,.0f}")
        print(f"  Value at Risk (1%):           ${m['VaR_1pct']:>12,.0f}")
        print(f"  Conditional VaR (5%):         ${m['CVaR_5pct']:>12,.0f}")

        print("DRAWDOWN ANALYSIS")
        print(f"  Median Max Drawdown:          {m['max_drawdown_median']:>8.1%}")
        print(f"  Mean Max Drawdown:            {m['max_drawdown_mean']:>8.1%}")
        print(f"  Worst 5% Drawdowns:           {m['max_drawdown_5th_percentile']:>8.1%}")
        print(f"  Single Worst Drawdown:        {m['max_drawdown_worst']:>8.1%}")
        print(f"  P(Drawdown > 20%):            {m['probability_of_20pct_drawdown']:>8.1%}")
        print(f"  P(Drawdown > 30%):            {m['probability_of_30pct_drawdown']:>8.1%}")
        print(f"  P(Drawdown > 50%):            {m['probability_of_50pct_drawdown']:>8.1%}")

        return m

    def print_scenario_results(self, scenarios):
        print("SCENARIO ANALYSIS")

        print(f"\n{'Scenario':<20} {'Median Value':>14} {'Median Return':>14} {'P(Loss)':>10} {'Med. DD':>10}")

        for name, s in scenarios.items():
            display_name = name.replace('_', ' ').title()
            print(
                f"{display_name:<20} ${s['terminal_median']:>12,.0f} {s['annualized_return_median']:>13.1%} {s['probability_of_loss']:>9.1%} {s['max_drawdown_median']:>9.1%}")

    def get_percentile_paths(self, percentiles=[5, 25, 50, 75, 95]):
        paths = {}
        for p in percentiles:
            paths[p] = np.percentile(self.simulation_paths, p, axis=0)
        return paths

    def export_results(self, filename="monte_carlo_results.json"):
        if self.metrics is None:
            self.calculate_metrics()

        percentile_paths = self.get_percentile_paths()

        results = {
            'parameters': {
                'initial_capital': self.initial_capital,
                'years': self.years,
                'num_simulations': self.num_simulations,
                'rebalance_frequency': self.rebalance_frequency,
                'risk_free_rate': self.risk_free_rate,
                'tickers': self.tickers,
                'weights': self.weights.to_dict(),
                'expected_returns': self.expected_returns.to_dict(),
            },
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in self.metrics.items()},
            'percentile_paths': {str(k): v.tolist() for k, v in percentile_paths.items()},
            'terminal_value_histogram': {
                'values': np.histogram(self.terminal_values, bins=50)[0].tolist(),
                'bin_edges': np.histogram(self.terminal_values, bins=50)[1].tolist(),
            }
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults exported to {filename}")
        return results


def run_simulation_from_optimizer(optimizer, years=10, num_simulations=10000, initial_capital=500000):
    simulator = MonteCarloSimulator(
        expected_returns=optimizer.mu_final,
        covariance_matrix=optimizer.cov_matrix,
        weights=optimizer.weights,
        initial_capital=initial_capital,
        years=years,
        num_simulations=num_simulations,
        rebalance_frequency='W',
        risk_free_rate=optimizer.risk_free_rate,
    )

    simulator.run_simulation()
    metrics = simulator.print_results()

    scenarios = simulator.run_scenario_analysis()
    simulator.print_scenario_results(scenarios)

    simulator.export_results("monte_carlo_results.json")

    return simulator, metrics