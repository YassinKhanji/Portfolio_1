## Signal Generation

The Signal Generation module forms the foundation of every strategy. It is responsible for identifying entry and exit signals using a combination of price action, indicators, and other techniques derived from technical, quantitative, or fundamental analysis.

### Point of Interest (POI)

A **Point of Interest (POI)** is a key price level where significant market activity is expected to occur, such as a reversal, consolidation, or breakout. This typically precedes the generation of an entry signal, serving as a precursor to actionable opportunities.

### Entry Signal

The **Entry Signal** defines the criteria for identifying bullish opportunities (as the strategy focuses exclusively on long positions). It acts as the primary trigger, signaling that a particular asset is ready for consideration for a long position.

### Specific Entry Point (SEP)

The **Specific Entry Point (SEP)** is the mechanism used to precisely time the entry into a trade. This could involve:
- A pullback to a predefined price level,
- Execution at a specific time of day, week, month, ...,
- A breakout above a certain price threshold, or similar events.

In some cases, the SEP may align directly with the entry signal, but it is designed to enhance precision in timing.

### Exit Signal

The **Exit Signal** outlines the criteria for bearish signals, which indicate the need to exit long positions. This serves as the primary method for closing trades, assuming neither the stop loss nor the take profit levels (tail risk management) have been triggered.