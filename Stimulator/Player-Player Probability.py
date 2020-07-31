d = {'Batsmen':[],'Bowler':[],'dot':[],'one':[],'two':[],'three':[],'four':[],'six':[],'wicket':[]}
fi = open('BatsmenBowler.csv','r')
head = 1
for lines in fi.readlines():
  line = lines.split(",")
  if(head == 1):
    head = 0
    continue
  batsmen = line[0]
  bowler = line[1]
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
  d['Bowler'].append(bowler)
  d['dot'].append(dot)
  d['one'].append(one)
  d['two'].append(two)
  d['three'].append(three)
  d['four'].append(four)
  d['six'].append(six)
  d['wicket'].append(wicket)

df = pd.DataFrame(d)
df.to_csv(r'Player-Player-prob', index=False) 