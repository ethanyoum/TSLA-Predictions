from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lag, when, lit, expr, to_date, avg, stddev

# Initialize Spark
spark = SparkSession.builder.appName("TSLA Alpha Signal Backtest").getOrCreate()

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(output_df)

# Ensure 'Date' column is datetime
spark_df = spark_df.withColumn("Date", F.to_date("Date"))

# Create a window spec for time series ops
window = Window.orderBy("Date")

# Generate signal
spark_df = spark_df.withColumn(
    "Signal",
    F.when(F.col("Pred_Prob") > 0.6, F.lit(1))
     .when(F.col("Pred_Prob") < 0.4, F.lit(-1))
     .otherwise(F.lit(0))
)

# Compute returns and strategy
spark_df = spark_df \
    .withColumn("Market_Return", F.col("Close") / F.lag("Close", 1).over(window) - 1) \
    .withColumn("Prev_Signal", F.lag("Signal", 1).over(window)) \
    .withColumn("Strategy_Return", F.col("Market_Return") * F.col("Prev_Signal"))

# Aggregate metrics
agg_df = spark_df.agg(
    F.avg("Strategy_Return").alias("Mean_Return"),
    F.stddev("Strategy_Return").alias("Volatility")
)

# Collect metrics
metrics = agg_df.collect()[0]
mean_return = metrics['Mean_Return']
volatility = metrics['Volatility']
sharpe_ratio = (mean_return / volatility) * (252**0.5) if volatility else None

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Mean Return: {mean_return:.5f}")
print(f"Volatility: {volatility:.5f}")


## Optimize Alpha Signal and Return
# Start Spark session
spark = SparkSession.builder.appName("TSLA Alpha Optimization").getOrCreate()

# Load the dataset
df = spark.read.csv("data_for_backtest.csv", header=True, inferSchema=True)
df = df.withColumn("Date", to_date("Date"))

# Define lag window
window = Window.orderBy("Date")

# Initialize best Sharpe tracker
best_sharpe = float("-inf")
best_upper = None
best_lower = None

# Loop through threshold pairs
for upper in np.arange(0.5, 0.9, 0.01):
    for lower in np.arange(0.1, 0.5, 0.01):
        if lower >= upper:
            continue

        # Generate signals
        temp_df = df.withColumn(
            "Signal",
            when(col("Pred_Prob") > upper, lit(1))
            .when(col("Pred_Prob") < lower, lit(-1))
            .otherwise(lit(0))
        )

        # Calculate returns
        temp_df = temp_df \
            .withColumn("Market_Return", col("Close") / lag("Close", 1).over(window) - 1) \
            .withColumn("Prev_Signal", lag("Signal", 1).over(window)) \
            .withColumn("Strategy_Return", col("Market_Return") * col("Prev_Signal"))

        # Aggregate metrics
        agg_df = temp_df.agg(
            avg("Strategy_Return").alias("Mean_Return"),
            stddev("Strategy_Return").alias("Volatility")
        )

        metrics = agg_df.collect()[0]
        mean_return = metrics["Mean_Return"]
        volatility = metrics["Volatility"]

        if volatility and volatility != 0:
            sharpe = (mean_return / volatility) * (252**0.5)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_upper = upper
                best_lower = lower

# Final output
print(f"Best Sharpe: {best_sharpe:.3f}")
print(f"Upper Threshold: {best_upper}, Lower Threshold: {best_lower}")
