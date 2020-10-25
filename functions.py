from pyspark.sql.functions import isnan, when, count, col, desc, \
    udf, split,countDistinct, to_date,avg,asc,round
from pyspark.sql.types import IntegerType, TimestampType
from pyspark.sql import SparkSession

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, \
    BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
 
def create_session():
    """Creates a spark sesion.

    Returns:
        obj: Spark session.
    """
    spark = SparkSession.builder.appName('sparkify').getOrCreate()
    return spark


def load_clean_data(data_path: str):
    """Cleans and transforms dataset

    Args:
        data_path (str): Path of the dataset.

    Returns:
        obj: Spark dataFrame.
    """    
    spark = create_session()
    sdf = spark.read.json(data_path)
    sdf = sdf.filter(sdf['auth'] != 'Guest')
    sdf = sdf.where(~((col('userId').isNull()) | (col('userId') == '')))
    churn_event = udf(lambda x: 1 if x == 'Cancellation Confirmation' else 0,\
         IntegerType())
    sdf = sdf.withColumn('churn', churn_event('page'))
    sdf = sdf.withColumn('state', split(sdf['location'], ',')[1])
    sdf = sdf.withColumn('date_time', (col('ts') / 1000.0)\
        .cast(TimestampType()))
    sdf = sdf.withColumn('date', to_date('date_time'))
    sdf = sdf.drop(*['firstName', 'lastName','location','ts'])
    return sdf


def features(sdf_clean):
    """Creates all features from the clean dataset.

    Args:
        sdf_clean (DataFrame): Spark DataFrame

    Returns:
        DataFrame: Spark DataFrame containing all created features.
    """
    churn_data = sdf_clean.select('userId', col('churn')).dropDuplicates()\
        .groupby('userId')\
        .agg({"churn": "max"}).withColumnRenamed("max(churn)", "label")

    gender = sdf_clean.select("userId", "gender").dropDuplicates()\
        .replace(['M', 'F'], ['0', '1'], 'gender').select('userId', \
            col('gender')\
            .cast('int'))

    friend_count = sdf_clean.where(sdf_clean.page == 'Add Friend')\
        .groupby('userId')\
        .agg(count('userId').alias('friend_count'))

    playlist_count = sdf_clean.where(sdf_clean.page == 'Add to Playlist')\
        .groupby('userId')\
        .agg(count('userId')\
        .alias('playlist_count'))

    song_count = sdf_clean.where(sdf_clean.page == 'NextSong')\
        .groupby('userId')\
        .agg(count('userId')\
        .alias('song_count'))

    thumbsup_count = sdf_clean.where(sdf_clean.page=='Thumbs Up')\
        .groupby("userId")\
        .agg(count('page')\
        .alias('thumbsup_count'))

    thumbsdown_count = sdf_clean.where(sdf_clean.page=='Thumbs Down')\
        .groupby("userId").agg(count('page')\
        .alias('thumbsdown_count'))

    avg_daily_sessions = sdf_clean.select('userId', 'date', 'sessionId')\
        .dropDuplicates()\
        .groupby('userId', 'date').agg(count('sessionId').alias('count'))\
        .groupby('userId').agg(round(avg('count')).alias('avg_daily_sessions'))
    
    features = churn_data.join(gender, "userId").join(friend_count, "userId")\
        .join(playlist_count, "userId")\
        .join(song_count, "userId").join(thumbsup_count, "userId")\
        .join(thumbsdown_count, "userId").join(avg_daily_sessions, "userId")

    return features

def feature_processing(features_df, columns, full_data=False):
    """Scales and prepares features  for fitting.

    Args:
        features_df (DataFrame): Spark DataFrame containing features.
        columns (list): List of column names.
        full_data (bool, optional): All data identifyer. Defaults to False.

    Returns:
        DataFrame|Multiple DataFrames: Returns training data.
    """
    # VectorAssembler
    assembler = VectorAssembler(inputCols=columns, outputCol="features")
    data = assembler.transform(features_df)

    # scale data
    scaler = StandardScaler(inputCol ='features', \
        outputCol ="final_features", withStd = True)
    scaler_model = scaler.fit(data)
    scale_data = scaler_model.transform(data)

    #Final features
    final_data = scale_data.select(scale_data.label,\
         scale_data.final_features.alias("features"))

    if full_data:
        return final_data
    else:
        train, test =final_data.randomSplit([0.80, 0.20], seed = 143)

        return train, test
        

def fit_model(data, classifier):
    """Fits model to training data.

    Args:
        data (DataFrame): Data to be fitted to model.
        classifier (obj): Classifier to fit data.

    Returns:
        obj: fitted model.
    """
    model_fit = classifier.fit(data)

    return model_fit


def test_model(model_fit, test_data):
    """Compares fitted model to test data.

    Args:
        model_fit (obj): Fitted model.
        test_data (DataFrame): Data to test the fitted model.
    """
    model_results = model_fit.transform(test_data)

    model_evaluation = MulticlassClassificationEvaluator(predictionCol\
        ='prediction')
    model_evaluation_auc = BinaryClassificationEvaluator(labelCol='label',\
         rawPredictionCol='prediction')

    print('Model Metrics')
    print('Accuracy: ', model_evaluation.evaluate(model_results, \
        {model_evaluation.metricName: 'accuracy'}))
    print('F-1 Score: ', model_evaluation.evaluate(model_results, \
        {model_evaluation.metricName: 'f1'}))
    print('Area under ROC Curve: ', model_evaluation_auc\
        .evaluate(model_results\
        , {model_evaluation_auc.metricName: 'areaUnderROC'}))
