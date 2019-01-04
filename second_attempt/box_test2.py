# instead of using measurements be some variation on the truth data, have unmoving boxes of some resolution as the measurements
from numpy import *
import matplotlib.pyplot as plt
from random import *
from mpl_toolkits.mplot3d import Axes3D

init_point = array([0,0,0])
del_x = 10
truth_adjust = array([del_x,randint(-10,10),randint(-10,10)])
#truth_adjust = array([del_x,3,0])
print truth_adjust
num_steps = 500
res =100
x_res = 2
def make_truth(init_point,truth_adjust):
	truth_datax =[init_point[0]]
	truth_datay =[init_point[1]]
	truth_dataz =[init_point[2]]
	new_point = init_point
	for t_step in range(num_steps):
		t=1
		new_point = new_point + dot(t,truth_adjust)
		truth_datax.append(new_point[0])
		truth_datay.append(new_point[1])
		truth_dataz.append(new_point[2])
	return (truth_datax,truth_datay,truth_dataz)
truth_x, truth_y,truth_z = make_truth(init_point,truth_adjust)

def make_measured(truth_x,truth_y,truth_z):
	measured_x =[]
	measured_y =[]
	measured_z =[]
	for x,y,z in zip(truth_x,truth_y,truth_z):
		m_x = x + randint(-x_res,x_res)
		m_y = y + randint(-res,res)
		m_z = z + randint(-res,res)
#		if m_x**2 + m_y**2 + m_z**2 < res**2:
#			continue
		measured_x.append(m_x)
		measured_y.append(m_y)
		measured_z.append(m_z)
	return measured_x,measured_y,measured_z
measured_x,measured_y,measured_z = make_measured(truth_x,truth_y,truth_z)

#def make_angle(x0,x1,z0,z1):
	#if (z1-z0) ==0:
	#	return 0
	
##############################################################
# make the state out of x, theta and m(slope)
current_state = array([measured_x[0],measured_y[0],measured_z[0], (measured_y[1]-measured_y[0])/del_x, (measured_z[1]-measured_z[0])/del_x])
print current_state
current_un = array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
state_history = [current_state]

F = array([[1,0,0,0,0],[0,1,0,del_x,0],[0,0,1,0,del_x],[0,0,0,1,0],[0,0,0,0,1]])
#B = array([1,0,0,0,])
u = del_x
Q = array([[x_res,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
C = array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
R = array([[res,0,0,0,0],[0,res,0,0,0],[0,0,res,0,0],[0,0,0,1,0],[0,0,0,0,1]])

def predict_step(x_cur,p_cur,step):
	B = array([1,0,0,0,0])#(x_cur[4]**3)/3])
	x_pre = dot(F,x_cur) + dot(B,u)
	p_pre = dot(dot(F,p_cur),transpose(F)) + Q
	return x_pre,p_pre

def update_step(step,x_cur,p_cur):
	x_pre, p_pre = predict_step(x_cur,p_cur,step)
	#z = array([measured_x[step],measured_y[step],measured_z[step]])
	z = array([measured_x[step],measured_y[step],measured_z[step], (measured_y[step]-measured_y[step-1])/del_x, (measured_z[step]-measured_z[step-1])/del_x])
	K = dot(dot(p_pre,transpose(C)), linalg.inv(dot(dot(C,p_pre),transpose(C)) + R))

	x_new = x_pre + dot(K,(z- dot(C,x_pre)))
	p_new = p_pre - dot(dot(K,C),p_pre)
	
	state_history.append(x_new)
	return x_new, p_new
for num in range(len(measured_x)-1):
	current_state, current_un = update_step(num+1,current_state,current_un)
kal_x = []
kal_y = []
kal_z = []
for i,state in enumerate(state_history):
	kal_x.append(state[0])
	kal_y.append(state[1])
	kal_z.append(state[2])
	if i < 100:
		print state[1], measured_y[i], truth_y[i]
##############################################################
fig=plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.plot(kal_x,kal_y,kal_z,c='g')
ax.plot(truth_x,truth_y,truth_z,c='r')
ax.plot(measured_x,measured_y,measured_z,c='b')
#ax.plot(kal_x,kal_y,kal_z,c='g')
plt.show()
ax = plt.subplot(111,projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.plot(kal_x,kal_y,kal_z,c='g')
ax.plot(truth_x,truth_y,truth_z,c='r')
#ax.plot(measured_x,measured_y,measured_z,c='b')
#ax.plot(kal_x,kal_y,kal_z,c='g')
plt.show()
ax = plt.subplot(111)
ax.set_xlabel("x")
ax.set_ylabel("dis")
ax.plot(truth_x, [sqrt( (kx-tx)**2 + (ky-ty)**2 + (kz-tz)**2) for kx,ky,kz,tx,ty,tz in zip(kal_x,kal_y,kal_z,truth_x,truth_y,truth_z)],'r')
ax.plot(truth_x, [(kx-tx) for kx,tx in zip(kal_x,truth_x)],'m')
ax.plot(truth_x, [(ky-ty) for ky,ty in zip(kal_y,truth_y)],'b')
ax.plot(truth_x, [(kz-tz) for kz,tz in zip(kal_z,truth_z)],'g')
plt.show()
print len(truth_x), len(kal_x)
