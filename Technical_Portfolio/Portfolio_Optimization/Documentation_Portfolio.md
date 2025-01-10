# Portfolio Management

The **portfolio management** module is responsible for the ongoing maintenance and optimization of the portfolio. It manages the rebalancing of asset allocations to achieve a specific target, such as maximizing returns, minimizing risk, or optimizing metrics like the Sharpe ratio. This module consists of two primary components: **Portfolio Optimization** and **portfolio Management**. It works closely with the testing or optimization module, which provides the methods necessary for optimization during execution. In live trading, these components should be called periodically to maintain and monitor the portfolio, portfolios, and individual trading systems.

## Portfolio Optimization

This component includes classes that handle various allocation strategies and rebalancing operations.

- **Allocation vs Rebalancing**: Allocation is the process of setting initial asset distributions based on market conditions, investor goals (e.g., growth, income generation, risk control, tax efficiency), and other factors. Rebalancing, on the other hand, involves periodically adjusting the allocations to maintain alignment with the target objectives.

- **Rebalance Types**: There are two primary types of rebalancing:
  1. **Target-Based Rebalancing**: Adjusting allocations to optimize for expected variables and achieve a predetermined target (e.g., 8% return).
  2. **Variable-Based Rebalancing**: Rebalancing based on historical data, assuming the continued relative differences in variables, to either minimize or maximize a target metric. This method is grounded in past performance and assumes historical trends will persist. The objective could involve minimizing or maximizing variables like returns, risk metrics, or performance ratios.

In other words, we could either optimize using expected variables or historical (trailing) variables. The objective function could be minimizing a variable, maximizing a varibale, and/or achievieng a value of a variable. To be precise the variables that we are refering to are metrics, whether it is sharpe-ratio, performance, Beta, Alpha, ...

## Portfolio Management

This component oversees Strategy-level management, ensuring that parameters like the maximum and minimum allocations per trade, as well as the number of trades, are adhered to. It also plays a key role in strategy monitoring for risk management, identifying when a strategy no longer aligns with the portfolio’s goals or becomes too risky to continue. Additionally, it handles optimization tasks, adjusting allocations and other sensitive parameters (e.g., indicator values).

Key metrics and factors considered include:

- **Market Regime**: Allocations are adjusted depending on whether the market is trending, mean-reverting, or in a transitional phase.
- **Asset Correlation**: Ensures diversification to minimize overconcentration in highly correlated assets or sectors, reducing risk.
- **Sensitivity Analysis**: Adjusts allocations and expectations based on how sensitive certain parameters are to market changes.

In summary, the portfolio managemenent module modifies the rebalanced by ensuring diversificiation, and matching strategies with their corresponding market conditions.

### Overconcentration Monitoring

The module monitors the exposure of individual assets or sectors, flagging instances of overconcentration that may increase risk. By enforcing limits on position sizes, it ensures that the portfolio maintains a balanced and diversified structure, avoiding the risks associated with excessive concentration in particular assets or sectors.


## Portfolio Risk Management

- Some ways to deal with Strategy Drawdowns in portfolio management:
	1. Moving Average X Price
	2. Detection of High autocorrelation for negative returns
	3. HMMs 
	4. Max Drawdown Stop Loss

Why perform the risk management on the overall portfolio, not on each strategy?

