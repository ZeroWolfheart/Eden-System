import numpy
import math

ss=0.2225

cd =numpy.zeros((5,4))

for a in range(0,5):
  cd[a] = [(ss-0.5)/2,
           (ss-0.1)/2,
           numpy.log(ss/2),
           numpy.log(ss/3)]
  print(cd[a])
  cd[a] = (cd[a]+5)/10
print(cd[a])       


print("asi sale el primero")
print (cd)

print("asi los otros")
for a in range(0,5):
  y=cd[a][0]
  x=cd[a][1]
  h=cd[a][2]
  w=cd[a][3]

  y= (y*10)-5
  x= (x*10)-5
  h= (h*10)-5
  w= (w*10)-5
  h= math.exp(h)
  w= math.exp(w)
  print(y,x,h,w)
