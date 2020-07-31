from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import concat, col, lit
from pyspark.sql.window import *
from pyspark.sql.functions import row_number
from pyspark import SparkContext
from copy import deepcopy
import numpy as np

spark = SparkSession\
    .builder\
    .appName("PythonBatnpspark")\
    .getOrCreate()\

batting = spark.read.csv('batting_new.csv', header=True, inferSchema=True)

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


t =[name.map(lambda x:(x, )).toDF(['name']),ave.map(lambda x:(x, )).toDF(['ave']),sr.map(lambda x:(x, )).toDF(['sr']),custom.map(lambda x:(x, )).toDF(['custom'])]

t[0]=t[0].withColumn('row_index', row_number().over(Window.orderBy(lit(1))))
t[1]=t[1].withColumn('row_index', row_number().over(Window.orderBy(lit(1))))
t[2]=t[2].withColumn('row_index',  row_number().over(Window.orderBy(lit(1))))
t[3]=t[3].withColumn('row_index',  row_number().over(Window.orderBy(lit(1))))

t1 = t[0].join(t[1],on = ["row_index"])
t2 = t1.join(t[2],on = ["row_index"])
t3 = t2.join(t[3],on = ["row_index"])
t4 = t3.drop("row_index")
t4.count()

vector = VectorAssembler(inputCols=['ave','sr','custom'], outputCol='cluster_features')
vector_fit = vector.transform(t4)
standardize = StandardScaler(inputCol='cluster_features', outputCol='standardized_features')
model = standardize.fit(vector_fit)
model_data = model.transform(vector_fit)

t5 = model_data.select('name','standardized_features')
def func(row):
  return (row.name,row.standardized_features)
all_b = t5.rdd.map(func)

X = np.array(all_b.collect())[:,1]

y = spark.sparkContext.parallelize(all_b.takeSample(withReplacement = False,num=4,seed = 4))

j = 0
r = np.zeros((4,3))
for i in y.collect():
  r[j] = np.array(i[1])
  j = j+1

def euc_dist(x,y, axis_=1):
    return (np.linalg.norm(np.array(x) - y, axis=axis_))

cluster_number =(np.zeros((3758,1)))

error = 1
ee = [x for x in range(3758)]
while error >0.05:
    for i in ee:
        d = spark.sparkContext.parallelize(euc_dist(X[i], r,1))
        cluster = np.argmin(np.array(d.collect()))
        cluster_number[i] = cluster
    old_r = deepcopy(r)
    for i in range(4):
        centroids = [X[j] for j in range(len(X)) if cluster_number[j] == i]
        r[i] = np.mean(centroids, axis=0)
    error = euc_dist(r, old_r,None)
    print(error)
