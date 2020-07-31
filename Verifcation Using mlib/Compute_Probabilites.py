import pandas as pd

d = {'Batsmen':[],'Batsmen_index':[],'Bowler':[],'Bowler_index':[],'dot':[],'one':[],'two':[],'three':[],'four':[],'six':[],'wicket':[]}
Bat_id = {}
Ball_id = {}
fi = open('BatsmenBowler.csv','r')
head = 1
bat_id = 0
ball_id = 0
for lines in fi.readlines():
  line = lines.split(",")
  if(head == 1):
    head = 0
    continue
  batsmen = line[0]
  bowler = line[1]
  if(batsmen not in Bat_id):
    Bat_id[batsmen] = bat_id
    bat_id = bat_id+1
  if(bowler not in Ball_id):
    Ball_id[bowler] = ball_id
    ball_id = ball_id+1
  balls = int(line[4]) + int(line[5]) + int(line[6])+ int(line[7])+ int(line[8])+ int(line[9])+ int(line[10])
  if(balls == 0):
    dot = one = two = three = four = six = wicket = 0
  else:
    dot = float(line[4])/balls
    one = float(line[5])/balls
    two = float(line[6])/balls
    three = float(line[7])/balls
    four = float(line[8])/balls
    six = float(line[9])/balls
    wicket = float(line[10])/balls
  d['Batsmen'].append(batsmen)
  d['Batsmen_index'].append(Bat_id[batsmen])
  d['Bowler'].append(bowler)
  d['Bowler_index'].append(Ball_id[bowler])
  d['dot'].append(dot)
  d['one'].append(one)
  d['two'].append(two)
  d['three'].append(three)
  d['four'].append(four)
  d['six'].append(six)
  d['wicket'].append(wicket)

df = pd.DataFrame(d)
df.to_csv(r'probablity.csv', index=False) 



##### Printing the ID For Batsmen or Bowler #######
d = {'Batsmen':[],'id':[]}
for i in Bat_id.keys():
  d['Batsmen'].append(i)
  d['id'].append(Bat_id[i])
df = pd.DataFrame(d)
df.to_csv(r'batsmen_id.csv', index=False)   

d = {'Bowler':[],'id':[]}
for i in Ball_id.keys():
  d['Bowler'].append(i)
  d['id'].append(Ball_id[i])
df = pd.DataFrame(d)
df.to_csv(r'bowler_id.csv', index=False)   