from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

# IMPORT OTHER MODULES HERE
import cleantext
import re
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from pyspark.sql.functions import concat, col, lit
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder,CrossValidatorModel
from pyspark.sql.functions import when

from pyspark.sql.functions import udf,col
from pyspark.sql.types import StringType,ArrayType,IntegerType,FloatType

def vec(st):
    return st.split(" ")


def main(context):
    """Main function takes a Spark SQL context."""
    # YOUR CODE HERE
    # YOU MAY ADD OTHER FUNCTIONS AS NEEDED
    #==============================# TASK 1 #===============================#

    #================= load the data and save ===============#
    #comments = sqlContext.read.json("data/comments-minimal.json.bz2")
    #comments.write.parquet("www/comments.parquet")
    
    #submissions = sqlContext.read.json("data/submissions.json.bz2")
    #submissions.write.parquet("www/submissions.parquet")

    #==============================# TASK 2 #===============================#
    # answers for questions are on report 

    #=========================# TASK 4 and TASK 5 #=========================#

    #============= get the cleaned text from clean_text.py =================#
    
    """
    comments = sqlContext.read.parquet("www/comments.parquet") # 10490094  
    comments.show()
    print ("===================================",comments.count())
    labeled = sqlContext.read.csv("www/labeled_data.csv",header=True)
    labeled = labeled.withColumnRenamed("Input_id","iid")
    labeled = labeled.withColumnRenamed("labeldem","Democrat")
    labeled = labeled.withColumnRenamed("labelgop","Republican")
    labeled = labeled.withColumnRenamed("labeldjt","Trump")
    labeled.show()
    print ("===================================",labeled.count())  #1775
    

    # # join labeled table and comments table
    
    comment_labeled = comments.join(labeled, labeled.iid == comments.id) #1781
    comment_labeled.show()
    print("===================================",comment_labeled.count())
    #comment_labeled.write.parquet("www/comment_labeled.parquet")
    """
    
    ## clean the text body  --udf
    """
    comment_labeled = sqlContext.read.parquet("www/comment_labeled.parquet")
    clean_udf = udf(cleantext.sanitize, StringType())
    vec_udf = udf(vec,ArrayType(StringType()))
    clean1 = comment_labeled.select('id',clean_udf("body").alias("cleaned"),'created_utc','author_flair_text',col('score').alias('com_score'),'link_id','Trump')
    comment_labeled_cleaned = clean1.select('id',vec_udf("cleaned").alias("words"),'created_utc','author_flair_text','com_score','link_id','Trump')

    print (comment_labeled_cleaned.count()) # 1781
    comment_labeled_cleaned.show()
    comment_labeled_cleaned.write.parquet("www/comment_labeled_cleaned.parquet")
    """
    
    #=========================# TASK 6 A B #=====================#

    #================= transfer to feature vector =================#
    """
    comment = sqlContext.read.parquet("www/comment_labeled_cleaned.parquet") 
    comment.show()
    cv = CountVectorizer(inputCol='words', outputCol="features", binary=True, minDF=5.0)

    model = cv.fit(comment)

    result = model.transform(comment)
    #result.show()
    #result.show(truncate=False)
    
    comments = result.select(result['id'],result['features'],result['Trump'])
    comments.show()
   
    comments = comments.withColumn("Trump", comments["Trump"].cast(IntegerType()))
    comments = comments.withColumnRenamed("Trump","label")
    
    pos = comments.withColumn("poslabel", when(comments["label"] == 1, 1).otherwise(0))
    neg = comments.withColumn("neglabel", when(comments["label"] == -1, 1).otherwise(0))
    pos.show()
    neg.show()
    
    #=========================# TASK 7 #=====================#
    
    #===================== training part ===================#
    # Initialize two logistic regression models.
    # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    poslr = LogisticRegression(labelCol="poslabel", featuresCol="features", maxIter=10).setThreshold(0.2)
    neglr = LogisticRegression(labelCol="neglabel", featuresCol="features", maxIter=10).setThreshold(0.25)
    
    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator()
    negEvaluator = BinaryClassificationEvaluator()
    
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,
        numFolds=5)
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=5)
        
    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50
    posTrain, posTest = pos.randomSplit([0.5, 0.5])
    negTrain, negTest = neg.randomSplit([0.5, 0.5])
    
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)

    # # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.save("www/pos.model")
    negModel.save("www/neg.model")
    """
    #===================== ROC score part ===================#

    """
    posresult = posModel.transform(posTest)   
    negresult = negModel.transform(negTest)
    

    pos_r = posresult.select(['probability', 'poslabel'])
    neg_r = negresult.select(['probability', 'neglabel'])

    pos_r.show()
    neg_r.show()

    pos_r_collect = pos_r.collect()
    pos_r_list = [(float(i[0][1]), float(i[1])) for i in pos_r_collect]
    posr = sqlContext.createDataFrame(pos_r_list)
    posr = posr.withColumnRenamed("_1","pos")
    posr = posr.withColumnRenamed("_2","poslabel")
    posr.show()

    neg_r_collect = neg_r.collect()
    neg_r_list = [(float(i[0][1]), float(i[1])) for i in neg_r_collect]
    negr = sqlContext.createDataFrame(neg_r_list)
    negr = negr.withColumnRenamed("_1","neg")
    negr = negr.withColumnRenamed("_2","neglabel")
    negr.show()

    posr.coalesce(1).write.option("header", "true").csv("pos_roc.csv")
    negr.coalesce(1).write.option("header", "true").csv("neg_roc.csv")

    scoreAndLabels = sc.parallelize(pos_r_list)
    metrics = metric(scoreAndLabels)
    print("The ROC score is for pos: ", metrics.areaUnderROC)

    scoreAndLabels = sc.parallelize(neg_r_list)
    metrics = metric(scoreAndLabels)
    print("The ROC score is for neg: ", metrics.areaUnderROC)

    exit()
    """
    #===================== ROC plot part ===================#

    #  on analysis.py

    #=========================# TASK 8 #=====================#

    #================ test part preprocessing ===============#
    """
    # pre-processing
    submissions = sqlContext.read.parquet("www/submissions.parquet")  #623073
    small_sub = submissions.select(col('id').alias('sub_id'),'title',col('score').alias('sub_score'))
    small_sub.show()
    
    comments = sqlContext.read.parquet("www/comments.parquet")
    comments.show()
    small_com = comments.select('id','body', 'link_id','created_utc','author_flair_text',col('score').alias('com_score'))
    small_com.show()
    
    # join submission
    com_sub_info = small_com.join(small_sub, small_com.link_id == concat(lit("t3_"),small_sub.sub_id))
    print (com_sub_info.count()) #10481025
    com_sub_info.show()

    # remove comments that contain /s or &gt
    com_sub_info=com_sub_info.select('id','body','title','created_utc','author_flair_text','com_score','sub_score').where(~col('body').like('%&gt;%'))
    com_sub_info=com_sub_info.select('id','body','title','created_utc','author_flair_text','com_score','sub_score').where(~col('body').like('%/s%'))
    com_sub_info.show()
    print (com_sub_info.count()) #9416981

    com_sub_info.write.parquet("www/com_sub_info.parquet")
    """
    
    #========================# TASK 9 #=======================#

    # =====================clean body text=====================
    """
    com_sub_info = sqlContext.read.parquet("www/com_sub_info.parquet")
    com_sub_info.show()
    
    clean_udf = udf(cleantext.sanitize, StringType())
    vec_udf = udf(vec,ArrayType(StringType()))
    
    clean1 = com_sub_info.select('id',clean_udf("body").alias("cleaned"),'title','created_utc','author_flair_text','com_score','sub_score')
    clean_text = clean1.select('id',vec_udf("cleaned").alias("words"),'title','created_utc','author_flair_text','com_score','sub_score')
    clean_text.show()
    print (clean_text.count())  #9417981
    clean_text.write.parquet("www/clean_text.parquet")
    """
    
    """
    comment = sqlContext.read.parquet("www/comment_labeled_cleaned.parquet") 
    cv = CountVectorizer(inputCol='words', outputCol="features", binary=True, minDF=5.0)
    model = cv.fit(comment)
   
    clean_text = sqlContext.read.parquet("www/clean_text.parquet")
    clean_text.show()

    result = model.transform(clean_text)
    #result = result.select('id','features')
    result.show()
    result.write.parquet("www/result_cv.parquet")
    """
    
    
    # ===================== model test =====================
    """
    result_cv = sqlContext.read.parquet("www/result_cv.parquet") 
    
    posModel=CrossValidatorModel.load("www/pos.model")
    posResult = posModel.transform(result_cv)
    posResult.show()
    
    posResult=posResult.select('id',col('prediction').alias("pos"),'title','created_utc','author_flair_text','com_score','sub_score')
    posResult.show()   
    posResult.write.parquet("www/posResult_full.parquet")
    """
    """
    result_cv = sqlContext.read.parquet("www/result_cv.parquet") 
    negModel=CrossValidatorModel.load("www/neg.model")
    negResult = negModel.transform(result_cv)
    negResult.show()

    negResult=negResult.select('id',col('prediction').alias("neg"),'title','created_utc','author_flair_text','com_score','sub_score')
    negResult.show()    
    negResult.write.parquet("www/negResult_full.parquet")
    """
    
    #======================# TASK 10 #======================#

    
    # ===================== task 10 (1) ====================
    """
    posResult = sqlContext.read.parquet("www/posResult_full.parquet")
    print ("=======================",posResult.count())  #9416981
    posResult.show()
    
    posNum = posResult.select('id','pos').where(col('pos')==1) #2757869
    print("=======================posNum=======================",posNum.count())
    
    
    negResult = sqlContext.read.parquet("www/negResult_full.parquet")
    print ("=======================",negResult.count())   #9416981
    negResult.show()
    negNum = negResult.select('id','neg').where(col('neg')==1)  #8369238
    print("=======================negNum=======================",negNum.count())
    """
    
    # ===================== task 10 (2) =====================
    """
    posResult = sqlContext.read.parquet("www/posResult_full.parquet")
    posResult.createOrReplaceTempView("posResult")
    posResult_b = sqlContext.sql("SELECT sum(pos)/count(pos) AS pos_percent,DATE(from_unixtime(created_utc)) AS date FROM posResult GROUP BY date ORDER BY date")
    posResult_b.show()
    #posResult_b.write.parquet("www/posResult_b.parquet")

    negResult = sqlContext.read.parquet("www/negResult_full.parquet")
    negResult.createOrReplaceTempView("negResult")
    negResult_b = sqlContext.sql("SELECT sum(neg)/count(neg) AS neg_percent,DATE(from_unixtime(created_utc)) AS date FROM negResult GROUP BY date ORDER BY date")
    negResult_b.show()
    #negResult_b.write.parquet("www/negResult_b.parquet")

    posResult_b = sqlContext.read.parquet("www/posResult_b.parquet")
    posResult_b.show()
    
    negResult_b = sqlContext.read.parquet("www/negResult_b.parquet")
    negResult_b.show()
    """
    
    # ===================== task 10 (3) =====================
    """
    posResult = sqlContext.read.parquet("www/posResult_full.parquet")
    posResult.createOrReplaceTempView("posResult")
    #posResult_c = sqlContext.sql("SELECT pos, author_flair_text FROM posResult WHERE author_flair_text IN ('Alabama', 'Alaska')")
    #states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
    posResult_c = sqlContext.sql("SELECT pos, author_flair_text FROM posResult WHERE author_flair_text IN ('Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming')")
    posResult_c.createOrReplaceTempView("posResult_c")
    posResult_c = sqlContext.sql("SELECT sum(pos)/count(pos) AS pos_percent,author_flair_text AS state FROM posResult_c GROUP BY state ORDER BY pos_percent DESC")
    
    posResult_c.show()
    posResult_c.write.parquet("www/posResult_c.parquet")
    """

    """
    negResult = sqlContext.read.parquet("www/negResult_full.parquet")
    negResult.createOrReplaceTempView("negResult")
    
    negResult_c = sqlContext.sql("SELECT neg, author_flair_text FROM negResult WHERE author_flair_text IN ('Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming')")
    negResult_c.createOrReplaceTempView("negResult_c")
    negResult_c = sqlContext.sql("SELECT sum(neg)/count(neg) AS neg_percent,author_flair_text AS state FROM negResult_c GROUP BY state ORDER BY neg_percent DESC")
    
    negResult_c.show()
    negResult_c.write.parquet("www/negResult_c.parquet")
    """
    
    # ===================== task 10 (4) =======================

    # -----------positive percent on comment/submissions score-------------
    """
    posResult = sqlContext.read.parquet("www/posResult_full.parquet")
    
    #comments = sqlContext.read.parquet("www/comments.parquet")  
    #submissions = sqlContext.read.parquet("www/submissions.parquet")  
    
    #small_posResult = posResult.select('id','pos')   
    #small_com = comments.select(col('id').alias('com_id'),col('link_id').alias('link_id'),col('score').alias('com_score'))
    #small_sub = submissions.select(col('id').alias('sub_id'),col('score').alias('sub_score'))
    
    #posResult_d_temp = posResult.join(small_com, posResult.id == small_com.com_id)    
    #posResult_d_temp2 = posResult_d_temp.join(small_sub, posResult_d_temp.link_id == concat(lit("t3_"),small_sub.sub_id))

    #posResult_d_temp2.write.parquet("www/posResult_d_temp2.parquet")

    posResult_d_temp2 = sqlContext.read.parquet("www/posResult_d_temp2.parquet")
    posResult_d_temp2.show()  
    posResult_d_temp2.createOrReplaceTempView("posResult_d_temp2")
    """
    
    """
    posResult = sqlContext.read.parquet("www/posResult_full.parquet")
    posResult.createOrReplaceTempView("posResult")

    posResult_d_com = sqlContext.sql("SELECT sum(pos)/count(pos) AS pos_percent, com_score FROM posResult GROUP BY com_score ORDER BY pos_percent DESC")
    #posResult_d_com.createOrReplaceTempView("posResult_d_com")
    posResult_d_com.show()
    #posResult_d_com.write.parquet("www/posResult_d_com.parquet")
    """
    """
    posResult = sqlContext.read.parquet("www/posResult_full.parquet")
    #posResult.show()
    posResult.createOrReplaceTempView("posResult") 
    #posResult_d_sub = sqlContext.sql("SELECT title,sub_score, sum(pos)/count(pos) AS pos_percent FROM posResult GROUP BY sub_score,title ORDER BY pos_percent DESC LIMIT 10")
    #posResult_d_sub.show(truncate=False)

    posResult_d_sub = sqlContext.sql("SELECT sum(pos)/count(pos) AS pos_percent, sub_score FROM posResult GROUP BY sub_score ORDER BY pos_percent DESC")
    posResult_d_sub.show()
    posResult_d_sub.createOrReplaceTempView("posResult_d_sub")
    posResult_d_sub.write.parquet("www/posResult_d_sub.parquet")
    """
    
    
    # -----------negative percent on comment/submissions score-------------
    
    """
    negResult = sqlContext.read.parquet("www/negResult_full.parquet")
    negResult.createOrReplaceTempView("negResult")
    
  
    negResult_d_com = sqlContext.sql("SELECT sum(neg)/count(neg) AS neg_percent, com_score FROM negResult GROUP BY com_score")
    negResult_d_com.createOrReplaceTempView("negResult_d_com")
    negResult_d_com.show()
    negResult_d_com.write.parquet("www/negResult_d_com.parquet")
    """
    
    """
    negResult = sqlContext.read.parquet("www/negResult_full.parquet")
    negResult.createOrReplaceTempView("negResult")
    
    negResult_d_sub = sqlContext.sql("SELECT title,sub_score, sum(neg)/count(neg) AS neg_percent FROM negResult GROUP BY sub_score,title ORDER BY neg_percent DESC LIMIT 10")
    
    
    negResult_d_sub = sqlContext.sql("SELECT sum(neg)/count(neg) AS neg_percent, sub_score FROM negResult GROUP BY sub_score")
    #negResult_d_sub.createOrReplaceTempView("negResult_d_sub")
    #negResult_d_sub.show(truncate=False)
    negResult_d_sub.write.parquet("www/negResult_d_sub.parquet")
    """
   
    
    """
    negResult_d_sub = sqlContext.read.parquet("www/negResult_d_sub.parquet")
    negResult_d_sub.show()  
    
    negResult_d_com = sqlContext.read.parquet("www/negResult_d_com.parquet")
    negResult_d_com.show()  
    """
    
    # ----------------write to csv to help make plots----------------
    """
    posResult_b = sqlContext.read.parquet("www/posResult_b.parquet")
    posResult_b.coalesce(1).write.option("header", "true").csv("www/2_Pos.csv")
    negResult_b = sqlContext.read.parquet("www/negResult_b.parquet")
    negResult_b.coalesce(1).write.option("header", "true").csv("www/2_Neg.csv")
    
    posResult_c = sqlContext.read.parquet("www/posResult_c.parquet")
    posResult_c.coalesce(1).write.option("header", "true").csv("www/3_Pos.csv")
    negResult_c = sqlContext.read.parquet("www/negResult_c.parquet")
    negResult_c.coalesce(1).write.option("header", "true").csv("www/3_Neg.csv")
    
    posResult_d_com = sqlContext.read.parquet("www/posResult_d_com.parquet")
    posResult_d_com.coalesce(1).write.option("header", "true").csv("www/4_Pos_com.csv")
    posResult_d_sub = sqlContext.read.parquet("www/posResult_d_sub.parquet")
    posResult_d_sub.coalesce(1).write.option("header", "true").csv("www/4_Pos_sub.csv")
    
    negResult_d_com = sqlContext.read.parquet("www/negResult_d_com.parquet")
    negResult_d_com.coalesce(1).write.option("header", "true").csv("www/4_Neg_com.csv")
    negResult_d_sub = sqlContext.read.parquet("www/negResult_d_sub.parquet")
    negResult_d_sub.coalesce(1).write.option("header", "true").csv("www/4_Neg_sub.csv")
    """
    
    # ===================== task 10 (5) ===========================

    """
    posResult = sqlContext.read.parquet("www/posResult_full.parquet")
    posResult.show()

    posResult.createOrReplaceTempView("posResult")
    posResult_e = sqlContext.sql("SELECT sum(pos)/count(pos) AS pos_percent FROM posResult WHERE title LIKE '%trump%'")
    #posResult_e = sqlContext.sql("SELECT sum(pos)/count(pos) AS pos_percent, title FROM posResult WHERE title LIKE '%trump%' GROUP BY title")
    posResult_e.show()

    negResult = sqlContext.read.parquet("www/negResult_full.parquet")
       
    negResult.createOrReplaceTempView("negResult")
    negResult_e = sqlContext.sql("SELECT sum(neg)/count(neg) AS neg_percent FROM negResult WHERE title LIKE '%trump%'")
    negResult_e.show()
    
    """
    

if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("www/cleantext.py")

    main(sqlContext)