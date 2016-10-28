# Topic - Linear regression using Stochastic gradient descent
# Author - Nandhini 
# Date - 10/27/2016
import sys
import datetime

#Dataset
x= [1,2,3,4,5]
y= [1,3,2,3,5]

'''
#Plot the data once to visualize
import matplotlib.pyplot as plt

plt.plot(x,y,"ro")
plt.axis([0,6,0,6])
plt.show()
'''

# The data points appear like they can be modelled using linear regression
# Consider the simple linear regression equation
# y=b0+b1*x
# where 
#		y is the dependent variable
#		x is the independent variable
#		b0 is the intercept
#		b1 is the slope
# For clarity, let us denote the actual value of y as y and the predicted value of y as p through out the code

# Problem definition
# Our aim here is to find the best possible combination of b0 & b1 so that the error between the actual and predicted value of y is as low as possible
# We will use the stochastic gradient descent approach that incrementally adjusts weights over every row
# The regular gradient descent runs through all the data before adjusting the weights every time

# Function definition
class SGD:

	def error_fn(self,p,y):
		err=p-y
		return err
		
	def predict(self,b0,b1,x):
		p=b0+(b1*x)
		return p
	
	def new_b0(self,b0,alpha,err):
		b0_new=b0-(alpha*err)
		return b0_new
	
	def new_b1(self,b1,aplha,err,x):
		b1_new=b1-(alpha*err*x)
		return b1_new
#Logic

# Begin by assiging random values for b0 & b1
b0=0
b1=0

# Assign learning rate alpha
alpha=0.02 

# Create object for class SGD
sgd_obj=SGD()

# Compute predicted y -> p incrementally
print datetime.datetime.now()
for j in range(0,4): # 4 is the number of epochs required. A epoch is one pass through the entire dataset. This is the condition for termination
	print "epoch: ",j
	print "starts: ", b0,b1
	p=[]
	for i in range(0,len(x)):
		print b0,b1
		p.append(b0+(b1*x[i]))
		err=sgd_obj.error_fn(p[i],y[i])
		#Simultaneously update b0 & b1 before using them in the next iteration
		b0_new=sgd_obj.new_b0(b0,alpha,err)	
		b1_new=sgd_obj.new_b1(b1,alpha,err,x[i])
		b0=b0_new
		b1=b1_new
	j+=1
print datetime.datetime.now()
	
# Create predictions using learned coefficients
final_p=[]
for i in range(0,len(x)):
	final_p.append(sgd_obj.predict(b0,b1,x[i]))

# Plot original y against predicted p
import matplotlib.pyplot as plt

plt.plot(x,y)
plt.plot(x,final_p)
plt.axis([0,6,0,6])
plt.show()
