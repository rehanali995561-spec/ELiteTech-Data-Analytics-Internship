from pyspark.sql import SparkSession
from pyspark.sql.functions import sum, avg, col

# Create Spark Session
spark = SparkSession.builder \
    .appName("Big Data Analysis") \
    .getOrCreate()

# Load Dataset
df = spark.read.csv("sales_data.csv", header=True, inferSchema=True)

# Show Dataset
print("Dataset Preview:")
df.show(5)

# Total Sales by Category
print("Total Sales by Category")
category_sales = df.groupBy("Category").agg(sum("Sales").alias("Total_Sales"))
category_sales.show()

# Average Sales by Category
print("Average Sales by Category")
avg_sales = df.groupBy("Category").agg(avg("Sales").alias("Average_Sales"))
avg_sales.show()

# Top 10 Highest Sales
print("Top 10 Highest Sales")
top_sales = df.orderBy(col("Sales").desc())
top_sales.show(10)

# Save Results
category_sales.write.mode("overwrite").csv("output/category_sales", header=True)
avg_sales.write.mode("overwrite").csv("output/average_sales", header=True)

print("Analysis Completed Successfully!")

spark.stop()
