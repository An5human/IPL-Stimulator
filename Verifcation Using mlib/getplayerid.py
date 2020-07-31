def get_bowler_id(bowler):
  fp = open(r'bowler_id.csv','r')
  ball = {}
  for i in fp.readlines():
    x = i.split(",")
    ball[x[0]] = int(x[1])
  return ball[batsmen]

def get_batsmen_id(batsmen):
  fp = open(r'batsmen_id.csv','r')
  bat = {}
  for i in fp.readlines():
    x = i.split(",")
    bat[x[0]] = int(x[1])
  return bat[batsmen]


def get_id(team_bat,team_ball):
      team_batid = []
      team_ballid = []
      for i in team_bat:
        x = get_batsmen_id(i)
        team_batid.append(x)
      for i in team_ball:
        x = get_bowler_id(i)
        team_ballid.append(x)
      return team_batid,team_ballid