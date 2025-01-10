# Portfolio Management

The **Portfolio Management** module is responsible for maintaining and optimizing the portfolio continuously. Its primary objective is to manage asset allocations to achieve specific targets, such as maximizing returns, minimizing risk, or optimizing metrics like the Sharpe ratio. This module is composed of three main components: **Portfolio Optimization**, **Portfolio Management**, and **Portfolio Risk Management**. It works in tandem with the testing or optimization module, which provides the necessary methods to support optimization during execution. In live trading, these components are executed periodically to monitor and maintain the portfolio, individual assets, and trading strategies.

---

## Portfolio Optimization

This component focuses on allocation strategies and rebalancing operations to align the portfolio with its goals.

### Allocation vs. Rebalancing

- **Allocation**: This involves setting initial asset distributions based on factors such as market conditions, investor goals (e.g., growth, income generation, risk control, tax efficiency), and other considerations.
- **Rebalancing**: Periodic adjustments to allocations ensure alignment with the portfolio’s target objectives.

### Types of Rebalancing

1. **Target-Based Rebalancing**: Adjusts allocations to meet predefined objectives, such as achieving an 8% return.
2. **Variable-Based Rebalancing**: Uses historical data to optimize allocations, assuming the persistence of relative differences in variables. This method focuses on minimizing or maximizing metrics like returns, risk levels, or performance ratios based on historical trends.

In essence, optimization can be driven by expected variables (forward-looking) or historical variables (trailing). The objective functions may include minimizing, maximizing, or achieving specific values for key metrics, such as the Sharpe ratio, performance, Beta, or Alpha.

---

## Portfolio Management

This component ensures strategy-level management and adherence to portfolio constraints, such as maximum and minimum allocations per trade or the number of trades. It monitors strategies to detect misalignment with portfolio goals or excessive risk. Additionally, it optimizes parameters, including allocations and indicator thresholds, to maintain performance.

Key factors considered include:

- **Market Regime**: Allocations are adjusted based on whether the market is trending, mean-reverting, or transitioning.
- **Asset Correlation**: Ensures diversification by limiting exposure to highly correlated assets or sectors, reducing overall risk.
- **Sensitivity Analysis**: Adjusts allocations based on the sensitivity of certain parameters to market changes.

In summary, the Portfolio Management component enforces diversification, aligns strategies with market conditions, and adjusts rebalancing strategies accordingly.

### Overconcentration Monitoring

The module monitors asset and sector exposures, identifying and flagging instances of overconcentration. By enforcing position size limits, it ensures a balanced and diversified portfolio, mitigating risks associated with excessive focus on specific assets or sectors.

---

## Portfolio Risk Management

This component addresses portfolio-level risks to ensure stability and resilience. Key strategies for handling drawdowns include:

1. **Moving Average vs. Price**: A technique to identify and react to unfavorable trends.
2. **High Autocorrelation Detection**: Recognizes prolonged periods of negative returns and their implications.
3. **Hidden Markov Models (HMMs)**: Used to identify and adapt to changing market states.
4. **Maximum Drawdown Stop Loss**: Limits losses by enforcing predefined drawdown thresholds.

### Why Focus on Portfolio-Level Risk Management?

A portfolio-wide approach to risk management offers several advantages:

1. **Elimination of Systemic Risk**: Individual strategy analysis may fail to address systemic issues if multiple strategies experience simultaneous drawdowns. A portfolio-level perspective captures these aggregated risks.
2. **Offsetting Losses**: Gains in one strategy may offset losses in another, which a portfolio-level approach accounts for effectively.
3. **Integration with Optimization**: Underperforming strategies are already addressed during portfolio optimization through reduced allocations, streamlining overall risk management.