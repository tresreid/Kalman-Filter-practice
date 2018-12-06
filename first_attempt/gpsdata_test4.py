# Adjusting the Control and noise matricies. Random acceleration is known but noise is included. Measurements taken for every 10 actual data points. 

from numpy import *
import matplotlib.pyplot as plt
from random import *
init_state = array([randint(100,200),randint(-10,10)])
del_t = .1
#accel = randint(-10,10)
num_steps = 1000
accel = [randint(-10,10) for x in range(num_steps)]
acc_noise = .1
#accel = [0]
#for step in range(num_steps):
#	accel.append(.1+accel[step])
#accel = [ 0 for x in range(num_steps/2)] + [ 5 for x in range(num_steps/2)]
fname = '../gpsdata/gpsdata/person1/0530/gpsdata01.txt'
x_coors = []
y_coors = []
times = []
truth_data1 = []
with open(fname) as f:
	lines = f.readlines()
	for line in lines:
		line.strip()
		lati, longi, time, x_proj, y_proj = line.split('|')
	
		hour, minute, second = time.strip().split(':')
#		print hour,minute, second
		if float(hour) < 0:
			hour = float(hour)+24
		new_time = 60*60*float(hour) + 60*float(minute) + float(second)
		times.append(new_time)
		x_coors.append(float(x_proj.strip()))
		y_coors.append(float(y_proj.strip()))	
		truth_data1.append(array([float(x_proj.strip()),float(y_proj.strip()),new_time]))	
#print truth_data1
#print times
measured_data = truth_data1
#def make_truth(x_0,v_0,a_0,steps):
#	truth_data1 = []
#	truth_data1all = []
#	pos= x_0
#	vel= v_0
#	for t_step in range(steps):
#		for tt_step in range(10):
#			t = del_t*.1
#			pos = pos + vel*t + 0.5*(a_0[t_step]+ acc_noise*randint(-100,100))*(t**2)
#			vel = vel + (a_0[t_step]+ acc_noise*randint(-100,100)) *t
#			truth_data1all.append(array([pos,vel]))
#		#pos = x_0 + v_0*t + 0.5*a_0[t_step]*(t**2)
#		#vel = v_0 + a_0[t_step]*t
#		truth_data1.append(array([pos,vel]))
#	return truth_data1, truth_data1all
#truth_data, truth_dataall = make_truth(init_state[0],init_state[1],accel,num_steps)
#print truth_dataall
#pos_variance = 10
#vel_variance = 10
#def make_measured(truth_data):
#	measured_data = []
#	for data in truth_data:
#		#measured_pos = gauss(data[0],.005)
#		#measured_vel = gauss(data[1],.005)
#		measured_pos = data[0] + randint(-pos_variance,pos_variance)
#		measured_vel = data[1] + randint(-vel_variance,vel_variance)
#		measured_data.append(array([measured_pos,measured_vel]))
#	return measured_data
##measured_data = make_measured(truth_data)
##measured_data = truth_data
#
####################
#
# Set initial state values, current state is initial measured values
#state_history =[measured_data[1]]
## Current state (curr x, previous x, curr velocity, curr time, previous time
current_state = array([measured_data[1][0],measured_data[0][0],(measured_data[1][0]-measured_data[0][0])/(times[1]-times[0]),0,0,times[1]])#,times[0]])
state_history =[current_state]
#print current_state
# prediction matrix: For this case, x = x_0 + vt, v = v+0
#F = array([[1,del_t],[0,1]])
# Our covariant matrix: Can start at one and becomes modified as the process runs
current_un = array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
# control matrix and vector: In this case, we add contribution from acceleration, x = x_0 + v0t + .5a0t^2, v= v0+a0t with constant acceleration vector(scalar)
# Takes into account things like commands issued by the moving object to slow down, speed up etc as well. "known" exactly how this factor is taken into account. Internal commands or known external forces etc?
#B = array([(0.5)*del_t**2,del_t])
#B = array([(0.5),1,1])
B = array([0,0,0,0,0,1])
u = times
### Q: Covariance of external noise (uncertainty from environment or perhaps uncertainty in "known" quanties like the control vectors)
#Q = array([[(100*acc_noise)*(0.5)*(del_t**2),0],[0,(100*acc_noise)*del_t]])
Q = array([[0,0],[0,0]])
### I think C (H) is just a matrix that makes the sensor reading compatible with our state readings. Just to change units? I'm not sure on this point. 
### It can also have more complicated affects. Perhaps the sensors can detect multiple state vector elements directly or indirectly. Ex) your state vector is position and velocity, but your sensor measures acceleration, so you derive the change in postion and velocity from that via the C(H) matrix
C = array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
### R: Covariance of the sensor noise (Distribution equal to the mean of the observed readings z)
#R = array([[pos_variance,0],[0,vel_variance]])
R = array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
#
## prediction step: predict next state by dotting the prediciton matrix with the current state and adding the control matrix (acceleration) prediciton
def predict_step(x_cur,p_cur,step):
#	print "xx",step, u[step],
	del_t = u[step-1]-u[step-2]
	#F = array([[1,del_t,0],[0,1,0],[0,0,1]])
	F = array([[1,0,del_t,0,del_t**2,0],[1,0,0,0,0,0],[0,0,1,0,del_t,0],[0,0,0,1,0,0],[0,0,(1/del_t),-(1/del_t),0,0],[0,0,0,0,0,0]])
	x_pre = dot(F,x_cur) + dot(B,u[step])

	p_pre = dot(dot(F,p_cur),transpose(F)) #+ Q
	return x_pre,p_pre


def update_step(step,x_cur,p_cur):
	x_pre, p_pre = predict_step(x_cur,p_cur,step)
	#z = measured_data[step]
	meas_v1 =(measured_data[step][0]-measured_data[step-1][0])/(times[step]-times[step-1])
	meas_v0 =(measured_data[step-1][0]-measured_data[step-2][0])/(times[step-1]-times[step-2])
	z = array([measured_data[step][0],measured_data[step-1][0],meas_v1,meas_v0, (meas_v1-meas_v0)/(times[step-1]-times[step-2]),times[step-1]])
	# Kalman gain
	G = dot(dot(p_pre,transpose(C)), linalg.inv(dot(dot(C,p_pre),transpose(C)) + R))
#	#K = dot(dot(p_pre,transpose(H)), linalg.inv(dot(dot(H,p_pre),transpose(H)) + R))
	
	#Updates to the new state and uncertainties
	x_new = x_pre +dot(G,(z - dot(C,x_pre)))
	p_new = p_pre - dot(dot(G,C),p_pre)
	state_history.append(x_new)
	return x_new,p_new
#
## Run all the updates
for num in range(len(times)-2):
	current_state, current_un = update_step(num+1,current_state,current_un)
#print state_history

# Get all values for easy plotting
truth_x=[]
#truth_v = []
truth_t = []
for point in truth_data1:
	truth_x.append(point[0])
	truth_t.append(point[2])
#truthall_x=[]
#truthall_v = []
#for point in truth_dataall:
#	truthall_x.append(point[0])
#	truthall_v.append(point[1])
#measured_x=[]
#measured_v = []
#for point in measured_data:
#	measured_x.append(point[0])
#	measured_v.append(point[1])
kal_x = []
#kal_v = []
kal_t = []

for point in state_history:
	kal_x.append(point[0])
	kal_t.append(point[5])
print "lengths t: ", len(truth_t), len(kal_t) 
#print [i for i in state_history if i[3] > 68140]

#
## Plot results
#
#ave_px = average([ (k-t) for k,t in zip(kal_x[1:],truth_x[1:])])
#ave_pv = average([ (k-t) for k,t in zip(kal_v[1:],truth_v[1:])])
#std_px = std([ (k-t) for k,t in zip(kal_x[1:],truth_x[1:])])
#std_pv = std([ (k-t) for k,t in zip(kal_v[1:],truth_v[1:])])
#std_mx = std([ (k-t) for k,t in zip(measured_x[1:],truth_x[1:])])
#std_mv = std([ (k-t) for k,t in zip(measured_v[1:],truth_v[1:])])
#print ave_px, std_px, std_mx
#print ave_pv, std_pv, std_mv
#
#print kal_t
#print truth_t
ax = plt.subplot(211)
#plt.plot([.1*x for x in range(num_steps*10)], truthall_x,"b-")
plt.plot( truth_t, truth_x,"b-")
plt.plot( kal_t, kal_x,"g-")
axr = plt.subplot(212)
plt.plot(kal_t, [ (k-t) for k,t in zip(kal_x,truth_x)],"g-")
#plt.plot(range(num_steps)[1:], [ (k-t) for k,t in zip(measured_x[1:],truth_x[1:])],"r-")
##plt.plot(range(num_steps)[1:], [1]*(num_steps-1),"r-")
plt.show()
#ax = plt.subplot(211)
#plt.plot([.1*x for x in range(num_steps*10)], truthall_v,"b-")
#plt.plot(range(num_steps), measured_v,"r-")
#plt.plot(range(num_steps), kal_v,"g-")
#axr = plt.subplot(212)
#plt.plot(range(num_steps)[1:], [ (k-t) for k,t in zip(kal_v[1:],truth_v[1:])],"g-")
#plt.plot(range(num_steps)[1:], [ (k-t) for k,t in zip(measured_v[1:],truth_v[1:])],"r-")
##plt.plot(range(num_steps)[1:], [1]*(num_steps-1),"r-")
#plt.show()
