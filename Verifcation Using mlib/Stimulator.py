player_prob = {}
fp = open('probability.csv','r')
head = 1            #To skip the header files
for i in f.readlines():
    if head == 1:
        head = 0
        continue
    x = i.split(",")
    batsmen = x[0]
    bowler = x[2]
    dot = x[4]
    one = x[5]
    two = x[6]
    three = x[7]
    four = x[8]
    six = x[9]
    wicket = x[10]
    player_prob[(batsmen,bowler)]= {'dot':dot,'one':one,'two':two,'three':three,'four':four,'six':six,'wicket':wicket}

colab_prob = {}
fc = open('collaborative_probabilty.csv','r')
head = 1            #To skip the header files
for i in f.readlines():
    if head == 1:
        head = 0
        continue
    x = i.split(",")
    batsmen = x[0]
    bowler = x[1]
    dot = x[2]
    one = x[3]
    two = x[4]
    three = x[5]
    four = x[6]
    six = x[7]
    wicket = x[8]
    colab_prob[(batsmen,bowler)] = {'dot':dot,'one':one,'two':two,'three':three,'four':four,'six':six,'wicket':wicket}


def probability(batsmen,bowler):
    if ((batsmen,bowler) in player_prob.keys()):
        return player_prob[(batsmen,bowler)]
    else:
        return colab_prob[(batsmen,bowler)]

def run_scored(batsmen,bowler):
    rand = random.random
    x = probability(batsmen,bowler)
    if(rand<x['dot']):
        return 0
    elif(rand<(x['dot']+x['one'])):
        return 1
    elif(rand<(x['dot']+x['one']+x['two'])):
        return 2
    elif(rand<(x['dot']+x['one']+x['two']+x['three'])):
        return 3
    elif(rand<(x['dot']+x['one']+x['two']+x['three']+x['four'])):
        return 4
    elif(rand<(x['dot']+x['one']+x['two']+x['three']+x['four']+x['six'])):
        return 6


def inning(team1,team2):
    '''
    The first team1 is batting and the second team is bowling.
    '''
    wicket = 0
    runs = 0
    striker = 0
    bowler = 0
    nonstriker = 1
    nextbat = 2
    wick_prob = {}
    out = 0
    for over in range(20):
        for ball in range(6):
            prob = probability(team1[striker],team2[bowler])
            wick_prob[(team1[striker],team2[bowler])] = wick_prob[(team1[striker],team2[bowler])]*prob['wicket']
            if(wick_prob[(team1[striker],team2[bowler])] < 0.5):
                out = 1
                striker = nextbat
                nextbat = nextbat +1
                print(team1[striker],"is out")
            if(not out):
                score = run_scored(team1[striker],team2[bowler])
                runs = runs + score
                if(score%2 == 1 and score < 4):
                    temp = striker
                    striker = nonstriker
                    nonstriker = striker
                print(team1[striker]," scored ",runs)
            wicket = wicket+out;
            if(wicket >= 10):
                print('Runs:',run,"Wickets",wickets,"Overs:",over,"Balls:",ball)   
                return runs,wickets
        bowler = (bowler + 1)%len(bowler)
        temp = striker
        striker = nonstriker
        nonstriker = striker
        return runs,wickets


if '__init__' == '__main__': 
	print("IPL Match Stimuator\n")
	print("Team1:\n")
	print(team1)
	print("Team2:\n")
	print(team2)
	print("1st Innings:\n")
	team1_run,team1_wicket = inning(team1,team2)
	print("2nd Innings:\n")
	team2_run,team2_wicket = inning(team2,team1)

	if(team1_runs > team2_runs):
    	print("Team1 Wins")
	elif(team1_runs < team2_runs):
    	print("Team2 Wins")
	else:
    	print("Its a tie")