import os
import pandas as pd


####Loading The Batsmen Data#######
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
                        six = 1 if x[7]=='6' else 0
                        four = 1 if x[7]=='4' else 0
                        wicket = 1 if len(x[9])>2 else 0
                        ball = 1
                        dot = 1 if x[7]=='0' and not wicket else 0
                        if(batsmen not in match_bb.keys()):
                            match_bb[batsmen] = [int(x[7]),four,six,wicket,1] #run,wicket,4's and 6's,balls faced    
                        else:    
                            match_bb[batsmen][0] += int(x[7])         #doesn't matter in case of wicket as it will be 0 
                            match_bb[batsmen][1] += four     
                            match_bb[batsmen][2] += six
                            match_bb[batsmen][3] = wicket            #wicket fallen
                            match_bb[batsmen][4] += ball                      
                f.close()

        for i in match_bb.keys():
          # Matches,Runs,4's,6's,Wickets,balls faced,50,100,0's , HighScore
          if(i not in bb.keys()):
            bb[i] = [1] + match_bb[i]
            if(match_bb[i][0]>=50):
              bb[i] = bb[i] + [1]
            else:
              bb[i] = bb[i] + [0]
            if(match_bb[i][0]>=100):
              bb[i] = bb[i] + [1]
            else:
              bb[i] = bb[i] + [0]
            if(match_bb[i][0]==0):
              bb[i] = bb[i] + [1]
            else:
              bb[i] = bb[i] + [0]
            bb[i] = bb[i] + [match_bb[i][0]]

          else:
            bb[i][0] = bb[i][0] + 1
            bb[i][1] = bb[i][1] + match_bb[i][0]
            bb[i][2] = bb[i][2] + match_bb[i][1]
            bb[i][3] = bb[i][3] + match_bb[i][2]
            bb[i][4] = bb[i][4] + match_bb[i][3]
            bb[i][5] = bb[i][5] + match_bb[i][4]
            if(match_bb[i][0]>=50):
              bb[i][6] = bb[i][6] + 1
            if(match_bb[i][0]>=100):
              bb[i][7] = bb[i][7] + 1
            if(match_bb[i][0]==0):
              bb[i][8] = bb[i][8] + 1
            if(bb[i][9] < match_bb[i][0]):
              bb[i][9] = match_bb[i][0]

d = {'Batsmen':[],'Matches':[],'Runs':[],'Wickets':[],'Balls':[],'0':[],'4':[],'6':[],'Strike Rate':[],'Average':[],'50':[],'100':[],'High Score':[]}
for i in bb.keys():
  d['Batsmen'].append(i)
  d['Matches'].append(bb[i][0])
  d['Runs'].append(bb[i][1])
  d['4'].append(bb[i][2])
  d['6'].append(bb[i][3])
  d['Wickets'].append(bb[i][4])
  d['Balls'].append(bb[i][5])
  d['50'].append(bb[i][6])
  d['100'].append(bb[i][7])
  d['0'].append(bb[i][8])
  d['High Score'].append(bb[i][9])
  d['Strike Rate'].append(float(bb[i][1])*100/int(bb[i][5]))
  if(int(bb[i][4])==0):
    d['Average'].append(float(bb[i][1])/int(bb[i][0]))
  else:
    d['Average'].append(float(bb[i][1])/int(bb[i][4]))
df = pd.DataFrame(d)
df.to_csv(r'/content/drive/My Drive/batsmen.csv', index=False)


####Loading The Bowler Data####### 
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
                        bowler = x[6]
                        six = 1 if x[7]=='6' else 0
                        four = 1 if x[7]=='4' else 0
                        wicket = 1 if len(x[9])>2 else 0
                        ball = 1
                        if(bowler not in match_bb.keys()):
                            match_bb[bowler] = [int(x[7]),four,six,wicket,1] #run,wicket,4's and 6's,balls faced    
                        else:    
                            match_bb[bowler][0] += int(x[7])         #doesn't matter in case of wicket as it will be 0 
                            match_bb[bowler][1] += four     
                            match_bb[bowler][2] += six
                            match_bb[bowler][3] += wicket            #wicket fallen
                            match_bb[bowler][4] += ball                      
                f.close()

        for i in match_bb.keys():
          # Matches,Runs,4's,6's,Wickets,balls faced,50,100,0's , HighScore
          if(i not in bb.keys()):
            bb[i] = [1] + match_bb[i]
            bb[i][5] = int(bb[i][5]/6) + int(bb[i][5]%6)  
          else:
            bb[i][0] = bb[i][0] + 1
            bb[i][1] = bb[i][1] + match_bb[i][0]
            bb[i][2] = bb[i][2] + match_bb[i][1]
            bb[i][3] = bb[i][3] + match_bb[i][2]
            bb[i][4] = bb[i][4] + match_bb[i][3]
            bb[i][5] = bb[i][5] + match_bb[i][4]   

d = {'Bowler':[],'Matches':[],'Overs':[],'Runs':[],'4':[],'6':[],'Wickets':[],'Economy':[],'Average':[],'Strike Rate':[]}
for i in bb.keys():
  d['Bowler'].append(i)
  d['Matches'].append(bb[i][0])
  d['Runs'].append(bb[i][1])
  d['4'].append(bb[i][2])
  d['6'].append(bb[i][3])
  d['Wickets'].append(bb[i][4])
  d['Overs'].append(int(bb[i][5]/6)+int(bb[i][5]%6)/10)
  if(int(bb[i][5]/6)==0):
    d['Economy'].append(float(bb[i][1])/int(bb[i][5]%6))
  else:
    d['Economy'].append(float(bb[i][1])/int(bb[i][5]/6))
  if(int(bb[i][4])==0):
    d['Average'].append(float(bb[i][1]))
  else:
    d['Average'].append(float(bb[i][1])/int(bb[i][4]))
  if(int(bb[i][4])==0):
    d['Strike Rate'].append(float(bb[i][5]))
  else:
    d['Strike Rate'].append(float(bb[i][5])/int(bb[i][4]))
df = pd.DataFrame(d)
df.to_csv(r'bowler.csv', index=False) 
