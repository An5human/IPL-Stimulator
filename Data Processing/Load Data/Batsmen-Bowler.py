import os
import pandas as pd

bb = {}
matches = 0
directory = "all_csv/"
for root,dirs,files in os.walk(directory):
    for file in files:
        if matches>5000:
          break
        match_bb = {}
        #1 match = 1CSV  |  Each match handling 
        if file.endswith(".csv"):
                matches += 1
                f = open(directory+file,'r') 
                for i in f.readlines():
                    x = i.split(",")
                    if(x[0]=="version"):
                        continue
                    elif(x[0]=='info'):
                        if(x[1]=='gender' and x[2]=='female'):
                            break
                    elif(x[0]=='ball'):
                        batsmen = x[4]
                        bowler = x[6]
                        pair = (batsmen,bowler) 
                        one = 1 if x[7]=='1' else 0
                        two = 1 if x[7]=='2' else 0
                        three = 1 if x[7]=='3' else 0
                        six = 1 if x[7]=='6' else 0
                        four = 1 if x[7]=='4' else 0
                        wicket = 1 if len(x[9])>2 else 0
                        #print(len(x[9]))
                        dot = 1 if x[7]=='0' and not wicket else 0 
                        ball = 1
                        if(pair not in match_bb.keys()):
                            match_bb[pair] = [int(x[7]),dot,one,two,three,four,six,wicket,ball] #run,wicket,4's and 6's    
                        else:    
                            match_bb[pair][0] += int(x[7])         #doesn't matter in case of wicket as it will be 0 
                            match_bb[pair][1] += dot
                            match_bb[pair][2] += one
                            match_bb[pair][3] += two
                            match_bb[pair][4] += three
                            match_bb[pair][5] += four     
                            match_bb[pair][6] += six
                            match_bb[pair][7] += wicket
                            match_bb[pair][8] += ball            #wicket fallen                     
                f.close()
        for i in match_bb.keys():
          if(i not in bb.keys()):
            bb[i] = [1] + match_bb[i]             #No of matches played by the combination
          else:
            bb[i][0] += 1
            bb[i][1] += match_bb[i][0]
            bb[i][2] += match_bb[i][1]
            bb[i][3] += match_bb[i][2]
            bb[i][4] += match_bb[i][3]
            bb[i][5] += match_bb[i][4]
            bb[i][6] += match_bb[i][5]
            bb[i][7] += match_bb[i][6]
            bb[i][8] += match_bb[i][7]
            bb[i][9] += match_bb[i][8]

d = {'Batsmen':[],'Bowler':[],'matches':[],'runs':[],'dot':[],'one':[],'two':[],'three':[],'four':[],'six':[],'wicket':[],'ball':[]}
for i in bb.keys():
  d['Batsmen'].append(i[0])
  d['Bowler'].append(i[1])
  d['matches'].append(bb[i][0])
  d['runs'].append(bb[i][1])
  d['dot'].append(bb[i][2])
  d['one'].append(bb[i][3])
  d['two'].append(bb[i][4])
  d['three'].append(bb[i][5])
  d['four'].append(bb[i][6])
  d['six'].append(bb[i][7])
  d['wicket'].append(bb[i][8])
  d['ball'].append(bb[i][9])

df = pd.DataFrame(d)
df.to_csv(r'BatsmenBowler.csv', index=False,header=False) 