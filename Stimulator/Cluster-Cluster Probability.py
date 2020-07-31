bat = {}
fb = open('Cluster_Data/batting_clusters.csv','r')
head = 1
for lines in fb.readlines():
  line = lines.split(",")
  if(head == 1):
    head = 0
    continue
  if line[1] not in bat:
    bat[line[1]] = line[2]

ball = {}
fb = open('Cluster_Data/bowler_clusters.csv','r')
head = 1
for lines in fb.readlines():
  line = lines.split(",")
  if(head == 1):
    head = 0
    continue
  if line[1] not in ball:
    ball[line[1]] = line[2]


cluster = {}
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
  
  if((bat[batsmen],ball[bowler]) not in cluster):
     cluster[(bat[batsmen],ball[bowler])] = [dot,one,two,three,four,six,wicket,1]
  else:
      cluster[(bat[batsmen],ball[bowler])][0] = cluster[(bat[batsmen],ball[bowler])][0] + dot
      cluster[(bat[batsmen],ball[bowler])][1] = cluster[(bat[batsmen],ball[bowler])][1] + one
      cluster[(bat[batsmen],ball[bowler])][2] = cluster[(bat[batsmen],ball[bowler])][2] + two
      cluster[(bat[batsmen],ball[bowler])][3] = cluster[(bat[batsmen],ball[bowler])][3] + three
      cluster[(bat[batsmen],ball[bowler])][4] = cluster[(bat[batsmen],ball[bowler])][4] + four
      cluster[(bat[batsmen],ball[bowler])][5] = cluster[(bat[batsmen],ball[bowler])][5] + six
      cluster[(bat[batsmen],ball[bowler])][6] = cluster[(bat[batsmen],ball[bowler])][6] + wicket
      cluster[(bat[batsmen],ball[bowler])][7] = cluster[(bat[batsmen],ball[bowler])][7] + 1

d = {'batting_clusters':[],'balling_clusters':[],'dot':[],'one':[],'two':[],'three':[],'four':[],'six':[],'wicket':[]}
for i in cluster.keys():
  d['batting_clusters'].append(i[0])
  d['balling_clusters'].append(i[1])
  d['dot'].append(cluster[i][0]/cluster[i][7])
  d['one'].append(cluster[i][1]/cluster[i][7])
  d['two'].append(cluster[i][2]/cluster[i][7])
  d['three'].append(cluster[i][3]/cluster[i][7])
  d['four'].append(cluster[i][4]/cluster[i][7])
  d['six'].append(cluster[i][5]/cluster[i][7])
  d['wicket'].append(cluster[i][6]/cluster[i][7])

df = pd.DataFrame(d)
df.to_csv(r'cluster-cluster-prob.csv', index=False)
