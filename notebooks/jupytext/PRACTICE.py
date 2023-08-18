# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jupytext//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# ### ICS20

import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import seaborn as sns
import scipy.io as sio

# #### Variables Data types(int, float, string, boolean)

integer = 1
print(type(integer))
float_number = 1.0
print(type(float_number))
string1 = "This is a string with double quotes"
print(type(string))
string = 'This is a string with single quotes'
print(type(string))
string2 = '''This is a string with three single quotes'''
print(type(string))
string = """This is a string with three double quotes"""
print(type(string))

print(string1)

print(string2)

# list
text = ["movie1","movie2", "movie3"]
numbers = [10,15,19,14,60]


# +

print(10 > 11)
print(10 < 11)
print(5 == 5)
print(5 != 5)
print(6 != 9)

# -

print(9//4)
print(9/4)

# Arithmetic Operators
x = 7
y = 3
print('x + y =',x + y)
print('x - y =',x - y)
print('x * y =',x * y)
print('x / y =',x / y)
print('x % y =',x % y)
print('x ** y =',x ** y)
print('x // y =',x // y)

# if statements
#
# ```python
# if condition1:
#     statement
#     statement
# else:
#     statement
# elif condition2 :
#     statement
# ```
#  Boolean Operator: 
# not , and , or 
#

p = 9
k = 9
if 9 == p and p == k:
    print("variable k and p are both equal to 9")


# practice
#  take a number `n` and print "odd" or "even" number
n = 84.5
if n%2 == 1 :
    print("odd")
elif n%2 ==0 :
    print("even")
else:
    print("you should enter integer!")

# Syntax
#
# ```python
# for target in range([start,] end[, increment]):
#     statement
#     [statements]
# ```
#
# `target`: variable that generate by `for`. Use this to reference in the function
#
# `range`: function that will generate a sequence of number base on the arguments. Usually use it with `for` to execute number of times
#
# `start`: starting value of generated range. This value is optional and default value is 0.
#
# `end`: value after end of the generated range, in other word up tp this value but not including. **required**
#
# `increment`: the different between each value in range. This value can only use if `start` and `end` are set in the range. This value is optional and default value is 1

# Skip increment
# output 0 2 4 6 8
for i in range(0,10,2):
    print(i)

for i in range(10):
    print(i)

# +
# Negative increment
# output : 9 8 7 6 5 4 3 2 1 0

for i in range(10,0,-1):
    print(i)

# +
# practice
#  take a number `n` and print all the even number between 0 to `n` inclusive.
n = 9
for i in range(0,n,2):
    print(i)

for i in range(n):
    if i%2 == 0:
        print(i)
# -

# Draws a rectangle with the character char
#     Parameters:
#         height: int
#         width: int
#         char: str

# print(4*"Arefeh")

print(4*"Arefeh")

# +
height = 9 ### should be integer
width = 4 ### should be integer
char = "*" ### one letter or one character

****
*  *

# +
height = 16 ### should be integer
width = 8 ### should be integer
char = "1234"

for h in range(height):
    if h==0 or h==height-1:
        print(width*char)
    else:
        print(char+(width-2)*" "+char)
# -


