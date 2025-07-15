# Time-Series Forecasting Models for TSLA stock prediction

https://medium.com/@ethanyoum981209/tesla-stock-analysis-prediction-modeling-95b219593664

## Introduction

Tesla has been one of the most influential and closely watched companies in the stock market, experiencing significant surges in value driven by key political, economic, and technological factors. Notably, Tesla’s stock saw substantial growth following the 2024 U.S. presidential election, where Donald Trump confirmed the victory over Kamala Harris. Investor optimism surged by nearly 15%, adding approximately $20 billion to Elon Musk’s net worth, as markets anticipated potential policy shifts favoring deregulation and changes to electric vehicle (EV) incentives.

This optimism was fueled by expectations of reduced regulatory oversight and potential adjustments to EV tax credits, which could strengthen Tesla’s market dominance and profitability. As a leading player in the EV and clean energy sector, Tesla’s stock price reflects not only its business fundamentals but also broader market sentiment, policy shifts, and technological advancements.

This report takes a data-driven approach to analyzing Tesla’s stock performance, financial indicators, and market trends. Additionally, it introduces predictive modeling techniques to assess potential future price movements based on historical data and key financial metrics, providing insights into Tesla’s investment risks and opportunities.

# Data Collection
Using Python’s Yahoo finance library, I gathered historical stock data spanning from December 14th, 2017, to December 14th, 2024, to assess Tesla’s financial health and performance over time. The dataset includes:

Stock Prices: Open, High, Low, Close, and Adjusted Close Prices.
Financial Ratios & Indicators: Price-to-Earnings (P/E), Price-to-Book (P/B), Annualized Volatility, and Classification to distinguish whether the price has been overestimated or underestimated.
Time Frame: Data covering recent 7 years for trend analysis.

<img width="697" height="505" alt="Screenshot 2025-07-15 at 1 21 58 AM" src="https://github.com/user-attachments/assets/03708d2c-0831-4bcc-a0bb-fc7571e5c2d5" />

# Financial Metrics / Trend Metrics Breakdown
Brief explanation about the metrics I used is:

Relative Strength Index (RSI): Assesses the speed and magnitude of the recent price.

P/E Ratio: Measures investor confidence and stock valuation.

P/B Ratio: Evaluates stock price relative to book value.

Annualized Volatility: Measures the degree of variation of the return over a given period.

# Quantitative Analysis About TSLA
<img width="680" height="58" alt="Screenshot 2025-07-15 at 1 23 39 AM" src="https://github.com/user-attachments/assets/808c3e69-cdbe-45ae-bb82-b5b2e82d4bff" />

To effectively analyze TSLA, I focused on key financial indicators, including the Relative Strength Index (RSI), Price-to-Earnings (P/E) Ratio, Price-to-Book (P/B) Ratio, and Annualized Volatility, to assess the stock’s valuation and market behavior.

The RSI value of 78.33 suggests that Tesla is currently in the overbought territory, indicating potential overvaluation and the possibility of a price correction. The P/E ratio of 107.75 is significantly higher than industry averages, reflecting strong investor expectations but also raising concerns about whether Tesla’s earnings can justify its current price. Furthermore, the P/B ratio of 20.01 suggests that Tesla’s stock price is trading at a substantial premium relative to its book value, which reinforces the overvaluation narrative.

Moreover, the Annualized Volatility of 0.63 highlights the stock’s inherent price fluctuations, suggesting a relatively high-risk investment. Given these indicators, Tesla is classified as “Overpriced”, meaning its current valuation may be driven more by speculative interest and market sentiment rather than fundamental financial performances.

To gain a deeper insights, I performed an Exploratory Data Analysis using the financial & trend metrics mentioned above.

<img width="969" height="517" alt="Screenshot 2025-07-15 at 1 25 24 AM" src="https://github.com/user-attachments/assets/be4e45e6-a979-48a0-b431-f42b3edbdd80" />

The shown plot is demonstrating Buy & Sell signals based on RSI and two types of Moving Averages, which are respectively fast and slow. The fast and slow moving averages provide insights into price momentum. The fast moving average reacts quickly to short-term price changes. whereas the slow moving average smooths fluctuations for long-term price changes. Buy and sell signals highlight optimal trading opportunities. Buy signals (green triangles) appear when the fast moving average crosses above the slow one, indicating upward momentum, while sell signals (red triangles) occur when it crosses below, showing a signal of the potential decline.
