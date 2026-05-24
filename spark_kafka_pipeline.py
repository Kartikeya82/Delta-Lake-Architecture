from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ======================
# SPARK SESSION
# ======================
spark = SparkSession.builder \
    .appName("Final Optimized ML Model") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ======================
# LOAD DATA
# ======================
df = spark.read.csv(
    "/home/puru/social_media_lakehouse/training.1600000.processed.noemoticon.csv",
    inferSchema=True,
    header=False
)

df = df.toDF("target", "id", "date", "flag", "user", "text")

print("\n=== SAMPLE DATA ===")
df.show(5)

# ======================
# LABEL CREATION
# ======================
df = df.withColumn(
    "label",
    when(col("target") == 4, 1).otherwise(0)
)

df = df.select("text", "label")

print("\n=== AFTER LABELING ===")
df.show(5)

# ======================
# FEATURE ENGINEERING (BEST VERSION)
# ======================
tokenizer = Tokenizer(inputCol="text", outputCol="words")

remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered"
)

vectorizer = CountVectorizer(
    inputCol="filtered",
    outputCol="rawFeatures",
    vocabSize=5000,
    minDF=5
)

idf = IDF(
    inputCol="rawFeatures",
    outputCol="features"
)

# ======================
# MODEL (BEST FOR TEXT)
# ======================
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20
)

pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, idf, lr])

# ======================
# TRAIN / TEST SPLIT
# ======================
train, test = df.randomSplit([0.8, 0.2], seed=42)

print("\nTraining count:", train.count())
print("Testing count:", test.count())

# ======================
# TRAIN MODEL
# ======================
print("\n🚀 Training optimized model...")
model = pipeline.fit(train)

# ======================
# PREDICTIONS
# ======================
predictions = model.transform(test)

print("\n=== PREDICTIONS SAMPLE ===")
predictions.select("text", "label", "prediction").show(10, truncate=False)

# ======================
# EVALUATION
# ======================
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

print("\n=== FINAL MODEL EVALUATION ===")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

# ======================
# SAVE MODEL
# ======================
model_path = "/home/puru/social_media_lakehouse/models/lr_model_final"

print("\n💾 Saving model to:", model_path)

model.write().overwrite().save(model_path)

print("\n✅ Final optimized model saved successfully!")
