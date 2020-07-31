from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from getplayerid.py  import *

spark = SparkSession\
		.builder\
		.appName("RecommenderSystems")\
		.getOrCreate()

# Load and parse the data
data = spark.read.text("probablity.csv").rdd.map(lambda r: r[0].split(','))
dot = data.map(lambda l: Rating(int(l[1]), int(l[3]), float(l[4])))
one = data.map(lambda l: Rating(int(l[1]), int(l[3]), float(l[5])))
two = data.map(lambda l: Rating(int(l[1]), int(l[3]), float(l[6])))
three = data.map(lambda l: Rating(int(l[1]), int(l[3]), float(l[7])))
four = data.map(lambda l: Rating(int(l[1]), int(l[3]), float(l[8])))
six = data.map(lambda l: Rating(int(l[1]), int(l[3]), float(l[9])))
wicket = data.map(lambda l: Rating(int(l[1]), int(l[3]), float(l[10])))


# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
model_dot = ALS.train(dot, rank, numIterations)
model_one = ALS.train(one, rank, numIterations)
model_two = ALS.train(two, rank, numIterations)
model_three = ALS.train(three, rank, numIterations)
model_four = ALS.train(four, rank, numIterations)
model_six = ALS.train(six, rank, numIterations)
model_wicket = ALS.train(wicket, rank, numIterations)

def probability(batsmen,bowler):
    dot =  model_dot.predict(batsmen,bowler)
    one = model_one.predict(batsmen,bowler)
    two = model_two.predict(batsmen,bowler)
    three = model_three.predict(batsmen,bowler)
    four = model_four.predict(batsmen,bowler)
    six = model_six.predict(batsmen,bowler)
    wicket = model_wicket.predict(batsmen,bowler)
    return {'dot':dot,'one':one,'two':two,'three':three,'four':four,'six':six,'wicket':wicket}


if '__init__' == '__main__':
	##Enter the teams
	team1 = []
	team2 = []

	team1_id,team2_id= get_id(team1,team2) 
	d = {'Batsmen':[],'Bowler':[],'dot':[],'one':[],'two':[],'three':[],'four':[],'six':[],'wicket':[]}
	bat_index = 0
	for i in team1_id:
		ball_index = 0
		for j in team2_id:
			x = probability(i,j)
			d['Batsmen'].append(team1[bat_index])
			d['Bowler'].append(team2[ball_index])
			d['dot'].append(x['dot'])
			d['one'].append(x['one'])
			d['two'].append(x['two'])
			d['three'].append(x['three'])
			d['four'].append(x['four'])
			d['six'].append(x['six'])
			d['wicket'].append(x['wicket'])
			ball_index = ball_index + 1
		bat_index = bat_index + 1
	df = pd.DataFrame(d)
	df.to_csv(r'collaborative_probabilty.csv', index=False)