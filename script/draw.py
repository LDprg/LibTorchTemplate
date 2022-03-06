import matplotlib.pyplot as plt
import sys

value = []
ai = []
sin = []  
  
for row in open(sys.argv[1],'r'):
    row = row.split(';')
    value.append(float(row[0]))
    ai.append(float(row[1]))
    sin.append(float(row[2]))

plt.title('AI')

plt.xlabel('Output')
plt.ylabel('Input')  

plt.plot(value, ai, marker = ' ', c = 'g')
plt.plot(value, sin, marker = ' ', c = 'b')
plt.show()