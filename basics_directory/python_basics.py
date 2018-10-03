
import os
def cls():
    os.system('cls' if os.name=='nt' else 'clear')
print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

print("This file path, relative to os.getcwd()")
print(__file__ + "\n")

print("This file full path (following symlinks)")
full_path = os.path.realpath(__file__)
print(full_path + "\n")

print("This file directory and name")
path, filename = os.path.split(full_path)
print(path + ' --> ' + filename + "\n")

print("This file directory only")
print(os.path.dirname(full_path))

#Print a message
print('Hello, world!')

#Print multiple values (of different types)
ndays = 365
print('There are', ndays, 'days in a year')

#Asking the user for a string
name = input('What is your name? ')

#Asking the user for a whole number (an integer)
num = int(input('Enter your age: '))

import datetime
now = datetime.datetime.now()
print (now.year, now.month, now.day, now.hour, now.minute, now.second)

print('Hi',name, 'You are', num, 'years old in', now.year)


#Decide to run a block (or not)
x = 3
if x == 3:
 print('x is 3')
 
#Are two values not equal?
print(x!=3)


#Less than another?
print(x<3)

#Greater than another?
print(x>3)

#Less than or equal to?
print(x <= 3)


#Greater than or equal to?
print(x >= 3)

#Decide between two blocks
mark = 80
if mark >= 50:
    print('pass')
else:
    print('fail')
#Decide between many blocks
mark = 80
if mark >= 65:
    print('credit')
elif mark >= 50:
    print('pass')
else:
    print('fail')

for i in range(10):
    print(i)

for i in range(0,10,1):
    print(i)


for i in range(10,-1,-1):
    print(i)

for c in 'Hello':
    print(c)

for c in 'Hello':
    print(c, end=' ')
print('!')  
  
#Compare two strings
msg = 'hello'
if msg == 'hello':
   print('howdy')
   
#Less than another string?  
if msg < 'n':
   print('a-m')
else:
   print('n-z')
   
#Is a character in a string?
if 'e' in msg:
    print("yes")
    
#Is a string in another string?
if 'ell' in msg:
    print("yes")
    
#Convert to uppercase
print(msg.upper())

#Count a character in a string
print(msg.count('l'))

#Replace a character or string
print(msg.replace('l','X'))


#Delete a character or string
print(msg.replace('l',''))


#Is the string all lowercase?
print(msg.islower())



##Repeat a block over list (or string) indices   
msg = 'I grok Python!'
for i in range(len(msg)):
    print(i, msg[i])


print(365 + 1 - 2,
25*9/5 + 32,
2**8)
print(str(365))
print(int('365'))

len('Hello')

print("credit",
'perfect',
'''Hello,
World!''')
print('Hello' + 'World')
print('Echo...'*4)
print(len('Hello'))
print(int('365'))


#Repeat a block 10 times
for i in range(10):
   print(i)
#Sum the numbers 0 to 9
   total = 0
for i in range(10):
    total = total + i
    print(total)
#Repeat a block over a string
for c in 'Hello':
    print(c)
#Keep printing on one line
for c in 'Hello':
    print(c, end=' ')
    print('!')


#Count from 1 to 10
for i in range(1,11):
   print(i)

#Count from 10 down to 1
for i in range(10,0,-1):
   print(i)

#Count 2 at a time to 10
for i in range(0,11,2):
   print(i)

#Count down 2 at a time
for i in range(10, 0, -2):
   print(i)

#Ask the user for a temperature in degrees Celsius
celsius = int(input('Temp. in Celsius: '))
#Calculate the conversion
fahrenheit = celsius*9/5 + 32
#Output the result
print(fahrenheit, 'Fahrenheit')

