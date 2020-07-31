# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

!tar xf '/content/drive/My Drive/hadoop_bd/spark-2.4.4-bin-hadoop2.7.tgz'
!pip install -q findspark

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.4.4-bin-hadoop2.7"

import findspark
findspark.init()
from pyspark.sql import SparkSession
#spark = SparkSession.builder.master("local[*]").getOrCreate()

from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import concat, col, lit
from pyspark.sql.window import *
from pyspark.sql.functions import row_number
from pyspark import SparkContext
from copy import deepcopy

spark = SparkSession\
    .builder\
    .appName("PythonMapReduceBat")\
    .getOrCreate()\

batting = spark.read.csv('/content/drive/My Drive/BD_FINAL/batting_new.csv', header=True, inferSchema=True)

#batting.show(3)

name =  batting.rdd.map(lambda r: r.Batsmen) 

matches = batting.rdd.map(lambda r: r.Matches)

runs = batting.rdd.map(lambda r: r.Runs)

out = batting.rdd.map(lambda r: r.Wickets)

balls = batting.rdd.map(lambda r: r.Balls)

high_score = batting.rdd.map(lambda r: r.High_Score)


ave = []
sr = []
custom = []

for i,j in zip(runs.collect(),out.collect()):
  if(j == 0):
    ave.append(float(0))
  else:
    ave.append(float(i/j))
    

for i,j in zip(runs.collect(),balls.collect()):
  if(j == 0):
    sr.append(float(0))
  else:
    sr.append(float(i/j) * 100)

for i,j,k in zip(matches.collect(),high_score.collect(),balls.collect()):
  if(k == 0):
    custom.append(float(0))
  else:
    custom.append(float((i * j) / k))



ave = spark.sparkContext.parallelize(ave)

sr = spark.sparkContext.parallelize(sr)

custom = spark.sparkContext.parallelize(custom)


t =[name.map(lambda x:(x, )).toDF(['Batsmen']),ave.map(lambda x:(x, )).toDF(['ave']),sr.map(lambda x:(x, )).toDF(['sr']),custom.map(lambda x:(x, )).toDF(['custom'])]

t[0]=t[0].withColumn('row_index', row_number().over(Window.orderBy(lit(1))))
t[1]=t[1].withColumn('row_index', row_number().over(Window.orderBy(lit(1))))
t[2]=t[2].withColumn('row_index',  row_number().over(Window.orderBy(lit(1))))
t[3]=t[3].withColumn('row_index',  row_number().over(Window.orderBy(lit(1))))

t1 = t[0].join(t[1],on = ["row_index"])
t2 = t1.join(t[2],on = ["row_index"])
t3 = t2.join(t[3],on = ["row_index"])
t4 = t3.drop("row_index")
t4.count()

#t4.show(4)

vector = VectorAssembler(inputCols=['ave','sr','custom'], outputCol='cluster_features')
vector_fit = vector.transform(t4)
standardize = StandardScaler(inputCol='cluster_features', outputCol='standardized_features')
model = standardize.fit(vector_fit)
model_data = model.transform(vector_fit)

t5 = model_data.select('Batsmen','standardized_features')
def func(row):
  return (row.Batsmen,row.standardized_features)
all_b = t5.rdd.map(func)

X = np.array(all_b.collect())
X = spark.sparkContext.parallelize(X)
y = spark.sparkContext.parallelize(all_b.takeSample(withReplacement = False,num=4,seed = 4))
#y.collect()

j = 0
r = np.zeros((4,3))
for i in y.collect():
  r[j] = np.array(i[1])
  j = j+1

def add_clusters(clusters,centroids):
  
    y = np.array(clusters[1:])
    x = np.array([float(y[0][0]),float(y[0][1]),float(y[0][2])])
    
    return (np.argmin(np.linalg.norm(x-centroids,axis = 1))),[clusters[0],np.array([float(y[0][0]),float(y[0][1]),float(y[0][2])])]

links = X.map(lambda clusters: add_clusters(clusters,r)).groupByKey()

def takeMean(points):
  io = np.array(points)  
  return np.mean(io[:,1],axis = 0)

ee = links.map(lambda cluster_points: (cluster_points[0],takeMean(list(cluster_points[1])) ) )
ee = ee.sortBy(lambda a: a[0])
kf = ee.map(lambda x: x[1]).collect()

links_new = X.map(lambda clusters: add_clusters(clusters,kf)).groupByKey()
ff = links_new.map(lambda cluster_points: (cluster_points[0],takeMean(list(cluster_points[1]))))
ff = ff.sortBy(lambda a: a[0])
gf = ff.map(lambda x: x[1]).collect()

error = np.linalg.norm(np.array(kf)- np.array(gf),None)

while error > 0.0005:
  kf = deepcopy(gf)
  
  links_new = X.map(lambda clusters: add_clusters(clusters,kf)).groupByKey()
  ff = links_new.map(lambda cluster_points: (cluster_points[0],takeMean(list(cluster_points[1]))))
  ff = ff.sortBy(lambda a: a[0])
  gf = ff.map(lambda x: x[1]).collect()
  print(error)
  error = np.linalg.norm(np.array(kf)-np.array(gf),axis  = None)

cluster_number = []
cluster_names = []
for i,j in links_new.collect():
  for k in j:
    cluster_number.append(i)
    cluster_names.append(k[0])

clus = spark.sparkContext.parallelize(cluster_number)
clus_names = spark.sparkContext.parallelize(cluster_names)
f = [clus_names.map(lambda x:(x, )).toDF(['Batsmen']),clus.map(lambda x:(int(x), )).toDF(['clus'])]

f[0]=f[0].withColumn('row_index', row_number().over(Window.orderBy(lit(1))))
f[1]=f[1].withColumn('row_index', row_number().over(Window.orderBy(lit(1))))

f1 = f[0].join(f[1],on = ["row_index"])
f2 = f1.drop('row_index')

f3 = f2.join(batting,on = ["Batsmen"])
f5 = f3.join(t4,on = ["Batsmen"])

f4 = f5.orderBy('_c0', ascending=True)

pandas_df = f4.select("*").toPandas()

pandas_df.to_csv('batting_clusters.csv')

#kf



