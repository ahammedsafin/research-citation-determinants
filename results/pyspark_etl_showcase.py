# PySpark ETL pipeline (requires Java + pyspark)
# pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length, regexp_replace, count, mean as spark_mean

spark = SparkSession.builder.appName("CitationETL").config("spark.driver.memory", "4g").getOrCreate()
sdf = spark.read.csv("dblp-v10.csv", header=True, inferSchema=True)
sdf = sdf.filter(col("year").between(2005, 2014))
sdf = sdf.withColumn("abstract_clean", regexp_replace(col("abstract"), r"[^a-zA-Z0-9 ]", " "))
sdf = sdf.withColumn("abstract_len", length(col("abstract_clean")))
venue_stats = sdf.groupBy("venue", "year").agg(
    count("*").alias("paper_count"), spark_mean("n_citation").alias("mean_citations"))
pdf = sdf.toPandas()
spark.stop()
