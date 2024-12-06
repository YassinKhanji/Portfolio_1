# Data

Under this module, data is managed by being called from the data source, cleaned, processed, prepared, and stored for future usage. This module ensures all essential data is readily available for subsequent phases.

### Log Returns vs Simple Returns

One of the main features to include when preparing the data is calculating the **log returns** and **simple returns** of the closing price of each trading instrument. Log returns offer several advantages over simple returns:
- **Enhanced Statistical Properties**: Metrics like the mean and standard deviation of log returns are more stable and interpretable.
- **Computational Efficiency**: Log returns simplify cumulative return calculations, as they can be added directly instead of using multiplicative operations.
- **Mathematical Symmetry**: Log returns handle gains and losses consistently, providing a balanced perspective for analysis.

By incorporating log returns, the analysis becomes more robust, efficient, and statistically sound.

### Coin Tickers as a Second Index

Having the coin tickers as a secondary index reduces the complexity of calculations by avoiding the need to loop through subcolumns. This structure streamlines operations, making it easier to filter, compute metrics, and apply conditions across multiple coins.