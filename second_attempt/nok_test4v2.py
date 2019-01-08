from numpy import *
import matplotlib.pyplot as plt
from random import *
from mpl_toolkits.mplot3d import Axes3D
from math import *
init_point = array([0,0,0])
del_x = 10
#truth_adjust = array([del_x,randint(-10,10),randint(-10,10)])
truth_adjust = array([del_x,0,1])
num_steps = 100
#res =5.0
#z_res = 1
BM = .0000001
rad = 1./BM # mv/qB = k/B let k=1
c = 3.0*10**(8)
a = 1/(c*BM)
res = 10
res_max = 10#(rad/10.)
#res_max = (rad/10.)
def make_truth(init_point,truth_adjust):
	truth_datax =[init_point[0]]
	truth_datay =[init_point[1]]
	truth_dataz =[init_point[2]]
	new_point = init_point
	phase_shift = pi
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
	for i,(x,y,z) in enumerate(zip(truth_x,truth_y,truth_z)):
		if i%1 ==0:
			m_x = x + (res_max/res)*randint(-res,res)
			m_y = y + (res_max/res)*randint(-res,res)
			m_z = z + (2./res)*randint(-res,res)
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
#current_state = array([measured_x[0],measured_y[0],measured_z[0], (measured_y[1]-measured_y[0])/del_x, (measured_z[1]-measured_z[0])/del_x])
init_azmu = atan((truth_y[1]-truth_y[0])/(truth_x[1]-truth_x[0]))
init_dip = abs((truth_z[1]-truth_z[0])/(sqrt((truth_x[1]-truth_x[0])**2+(truth_y[1]-truth_x[0])**2)))
init_dz = truth_z[1]-measured_z[1]
init_dp = (sqrt((truth_x[1]-truth_x[0])**2+(truth_y[1]-truth_x[0])**2) - (sqrt((measured_x[1]-measured_x[0])**2+(measured_y[1]-measured_x[0])**2)))
current_state = array([init_dp,init_azmu,init_dz,init_dip])
current_un = array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
state_history = [current_state]
state_history_coords = [array([0,0,0])]

#F = array([[1,0,0,0,0],[0,1,0,del_x,0],[0,0,1,0,del_x],[0,0,0,1,0],[0,0,0,0,1]])
#B = array([0,0,0,0,0])
u = del_x
C = array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
R = array([[res_max**2,0,0],[0,res_max**2,0],[0,0,.2**2]])
def make_coords(state,con,meas):
	con=1
	x_coord = meas[0] + state[0]*cos(state[1]) - con*sin(state[1])
	y_coord = meas[1] + state[0]*sin(state[1]) + con*cos(state[1])
	z_coord = meas[2] + state[2] +(state[3])*con
	#print array([x_coord,y_coord,z_coord])
	print state
	return array([x_coord,y_coord,z_coord])
def predict_step(x_cur,p_cur,step,del_phi,dak0,dak1,k,tanl):
	#k = x_cur[2]
	#tanl = x_cur[4]
	#phi0 = x_cur[1]

	#dak1 = (dp1+a/k)
	#dak0 = (dp0+a/k)
	#del_phi = phi1-phi0
	
	F = array([ 
	[cos(del_phi),1000*sin(del_phi),0,0], #d dP/da
	[0,cos(del_phi),0,0], # dphi/da
	[tanl*sin(del_phi),tanl*(1-cos(del_phi)),1,1], #ddz/da
	[0,0,0,1] #dtanl/da
	])
	B = array([0,0,0,0])#(x_cur[4]**3)/3])
	#print "tanl: ",tanl
	suppression = .0001
	Q = suppression*array([
	[0,0,0,0,0],
	[0,(1+(tanl*tanl)),0,0,0],#[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
	[0,0,(k*tanl)*(k*tanl),0,k*tanl*(1+(tanl*tanl))],#[0,0,0,0,0],[0,0,0,0,0]])
	[0,0,0,0,0],
	[0,0,k*tanl*(1+(tanl*tanl)),0,(1+(tanl*tanl))*(1+(tanl*tanl))]])

	x_pre = dot(F,x_cur) + dot(B,u)
	p_pre = dot(dot(F,p_cur),transpose(F))# + Q
	return x_pre,p_pre

def update_step(step,x_cur,p_cur):
	#x_pre, p_pre = predict_step(x_cur,p_cur,step)
	phi0 = x_cur[1]
	tanl = x_cur[3]
	k =1# x_cur[2]
	dp0 = x_cur[0]
	dak0 = (dp0+a/k)
	if measured_x[step]-measured_x[step-1] == 0:
		phix = pi/2
	else:
		phix= atan((measured_y[step]-measured_y[step-1])/(measured_x[step]-measured_x[step-1]))
		#phix = (2*pi/(2*num_steps))
	z = array([measured_x[step],measured_y[step],measured_z[step]])# (measured_y[step]-measured_y[step-1])/del_x, (measured_z[step]-measured_z[step-1])/del_x])

	#H = array([
	#[cos(phi0),-dak0*sin(phi0)+(a/k)*sin(phi0+phix),-(a/(k*k))*(cos(phi0)-(cos(phi0+phix))),0,0],
	#[sin(phi0),dak0*cos(phi0)-(a/k)*cos(phi0+phix),-(a/(k*k))*(sin(phi0)-(sin(phi0+phix))),0,0],
	#[0,0,(a/(k*k))*phix*tanl,1,-a*phix/k],
	#])
	con=1#-1*measured_z[step]
	H = array([
	[cos(phi0),-dp0*sin(phi0)-con*cos(phi0),0,0],
	[sin(phi0),dp0*cos(phi0)- con*sin(phi0),0,0],
	[0,0,1,con],
	])

	#z = array([measured_x[step],measured_y[step],measured_z[step]])
	
	z_state = dot(z,H)
	#print z_state
	dp1 = z_state[0]
	phi1 = z_state[1]
	dak1 = (dp1+a/k)
	del_phi = phi1-phi0
	
	x_pre, p_pre = predict_step(x_cur,p_cur,step, del_phi,dak0,dak1,k,tanl)


	K = dot(dot(p_pre,transpose(H)), linalg.inv(dot(dot(H,p_pre),transpose(H)) + R))

	x_new = x_pre + dot(K,(z- dot(H,x_pre)))
	p_new = p_pre - dot(dot(K,H),p_pre)
	
	z0 = array([measured_x[step-1],measured_y[step-1],measured_z[step-1]])# (measured_y[step]-measured_y[step-1])/del_x, (measured_z[step]-measured_z[step-1])/del_x])
	state_history.append(x_new)
	state_history_coords.append(make_coords(x_new,con,z))#state_history_coords[-1]))
	return x_new, p_new
for num in range(len(measured_x)-1):
	current_state, current_un = update_step(num+1,current_state,current_un)
#	print current_state
kal_x = []
kal_y = []
kal_z = []
for i,state in enumerate(state_history_coords):
	kal_x.append(state[0])
	kal_y.append(state[1])
	kal_z.append(state[2])
	#if i < 100:
	#	print state_history_coords[i]
##############################################################
fig=plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
#ax.plot(kal_x,kal_y,kal_z,c='g')
ax.plot(truth_x,truth_y,truth_z,c='r')
ax.plot(measured_x,measured_y,measured_z,c='b')
ax.plot(kal_x,kal_y,kal_z,c='g')
plt.show()
#ax = plt.subplot(111,projection='3d')
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.set_zlabel("z")
#ax.plot(kal_x,kal_y,kal_z,c='g')
#ax.plot(truth_x,truth_y,truth_z,c='r')
#plt.show()
ax = plt.subplot(111)
ax.set_xlabel("x")
ax.set_ylabel("dis")
#ax.plot(truth_x, [sqrt( (kx-tx)**2 + (ky-ty)**2 + (kz-tz)**2) for kx,ky,kz,tx,ty,tz in zip(kal_x,kal_y,kal_z,truth_x,truth_y,truth_z)],'r')
ax.plot(truth_x, [(kx-tx) for kx,tx in zip(kal_z,truth_z)],'m')
ax.plot(truth_x, [(ky-ty) for ky,ty in zip(measured_z,truth_z)],'b')
#ax.plot(truth_x, [(kz-tz) for kz,tz in zip(kal_z,measured_z)],'g')
print [(kx-tx) for kx,tx in zip(kal_z,measured_z)]
plt.show()
print state_history[0], measured_x[0],measured_x[1],measured_y[0],measured_y[1],measured_z[0],measured_z[1]