from numpy import *
import matplotlib.pyplot as plt
from random import *
from mpl_toolkits.mplot3d import Axes3D
from math import *
init_point = array([0,0,0])
del_x = 10
truth_adjust = array([del_x,randint(-10,10),randint(-10,10)])
#truth_adjust = array([del_x,3,0])
num_steps = 1000
#res =5.0
#z_res = 1
#rad = 1./BM # mv/qB = k/B let k=1
#c = 3.0*10**(8)
#BM = 10./c
a = 10000.#1/(c*BM)
k_0=1.
rad = a/k_0#1./BM # mv/qB = k/B let k=1
res = 10
res_max = 30
ang_steps = 2*pi/(1.1*num_steps)
z_step = 60
def get_angle(meas1):
	m_phi = atan((meas1[1])/(meas1[0]))
	if sign(meas1[0]) <0: # only if x is negative add pi
		m_phi = m_phi +pi
	if sign(meas1[0]) >0 and sign(meas1[1])<0:
		m_phi = m_phi +2*pi
	if meas1[2] > 10*5*(num_steps/6) and m_phi<pi/6:
		m_phi = m_phi+2*pi
#	if meas1[2] < 300 and m_phi>(5*pi/6):
#		m_phi = m_phi-2*pi
	m_phi = m_phi%(2*pi)
	return m_phi
def make_truth(init_point,truth_adjust):
	truth_datax =[]
	truth_datay =[]
	truth_dataz =[]
#	new_point = init_point
#	phase_shift = pi
	for t_step in range(num_steps):
		#t=1
		#new_point = new_point + dot(t,truth_adjust)
		truth_datax.append(rad*(cos((t_step*ang_steps))))
		truth_datay.append(rad*(sin(t_step*ang_steps)))
		truth_dataz.append(z_step*t_step)
	return (truth_datax,truth_datay,truth_dataz)
truth_x, truth_y,truth_z = make_truth(init_point,truth_adjust)

def make_measured(truth_x,truth_y,truth_z):
	measured_tuple =[]
	for i,(x,y,z) in enumerate(zip(truth_x,truth_y,truth_z)):
		if i%1 ==0:
			m_x = x + (res_max/res)*randint(-res,res)
			m_y = y + (res_max/res)*randint(-res,res)
			m_z = z + (res_max/res)*randint(-res,res)
			measured_tuple.append((m_x,m_y,m_z))
	#measured_sorted = measured_tuple
	measured_sorted = sorted(measured_tuple, key=lambda x: x[2])
	#measured_sorted = sorted(measured_tuple, key=lambda x: get_angle(x))
	return measured_sorted
measured_all = make_measured(truth_x,truth_y,truth_z)
measured_x =[]
measured_y =[]
measured_z =[]
for meas_x,meas_y,meas_z in measured_all:
	measured_x.append(meas_x)
	measured_y.append(meas_y)
	measured_z.append(meas_z)

def make_coords(state,phix,meas,z =False,dz0=0):
	x_coord = state[0]*cos(state[1]) + (a/(state[2]))*(cos(state[1]))
	y_coord = state[0]*(sin(state[1])) + (a/(state[2]))*(sin(state[1]))
	z_coord = meas[2] + (a/(state[2]))*(state[4])*phix - state[3] +dz0
#	z_coord =  (a/(state[2]))*(state[4])*state[1] - state[3] #+dz0
#	if z:
#		phix = ang_steps
#		z_coord =  (a/(state[2]))*(state[4])*state[1] #- state[3] #+dz0
#		#z_coord = meas[2] + (a/(state[2]))*(state[4])*2*abs(sin(phix/2)) + state[3] -dz0
#		z_coord = meas[2] + (a/(state[2]))*(state[4])*phix - state[3] +dz0
	return array([x_coord,y_coord,z_coord])
def convert_meas(meas0,meas1,xc,z=False,step=0):
	###still have issues with X_c and Y_c. Also dp does not always give good values for truth
#	X_c = meas0[0] + (xc[0]-a/xc[2])*abs(cos(xc[1]))
#	Y_c = meas0[1] + (xc[0]-a/xc[2])*abs(sin(xc[1]))
	X_c =0 
	Y_c =0 
#	print "X_c: ", X_c
#	print "Y_c: ",Y_c	
	m_phi = get_angle(meas1)
	m_dp = (X_c-meas1[0])*cos(m_phi) + (Y_c-meas1[1])*sin(m_phi) + (a/xc[2])
	m_k = xc[2]
	m_tanl = xc[4]
	m_dz = (meas0[2]-meas1[2])+((a/xc[2])*(m_phi-xc[1])*m_tanl) + xc[3]
#	if z:
#		m_dz = truth_z[step]-measured_z[step]	
#		m_sgn=(sqrt((truth_x[step])**2+(truth_y[step])**2) - (sqrt((measured_x[step])**2+(measured_y[step])**2)))
#		m_dp =sqrt( measured_x[step]**2 + measured_y[step]**2) - (a/m_k) 
#	#	m_tanl = (meas1[2]-meas0[2])/(sqrt((meas1[0]-meas0[0])**2+(meas1[1]-meas0[1])**2))
##		m_tanl = (meas1[2]-0)/(sqrt((meas1[0]-0)**2+(meas1[1]-0)**2))
#		m_tanl = (meas1[2]-0)/(m_phi*a/m_k)
##		m_tanl = (truth_z[step]-0)/(rad*ang_steps*step)
	return array([m_dp,m_phi,m_k,m_dz,m_tanl])
##############################################################
# make the state out of x, theta and m(slope)
#init_azmu = atan((truth_y[0])/(truth_x[0]))
#init_dip = (truth_z[1]-truth_z[0])/(sqrt((truth_x[1]-truth_x[0])**2+(truth_y[1]-truth_y[0])**2))
#init_dz = truth_z[1]-measured_z[1]
#init_dp =sqrt( measured_x[0]**2 + measured_y[0]**2) - (a/k_0) 
z0 = array([measured_x[0],measured_y[0],measured_z[0]])
z1 = array([measured_x[1],measured_y[1],measured_z[1]])
#print "meas: "
init_dp,init_azmu,init_k,init_dz,init_dip = convert_meas(z0,z1,array([0,0,1,0,z_step/(2*pi*a/ang_steps)]),True,1)	

init_dip_t = (truth_z[1]-0)/(rad*ang_steps)
current_state = array([init_dp,init_azmu,k_0,init_dz,init_dip])
#current_state = array([100,.002,-4*k_0,1,.2])
#current_state = [array([0,0,1,0,init_dip_t])]
current_un = array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
state_history = [current_state]
state_history_coords = [array([rad,0,0])]
state_history_coords_true = [array([truth_x[0],truth_y[0],truth_z[0]])]
state_history_coords_meas = [array([measured_x[0],measured_y[0],measured_z[0]])]
state_history_true = [array([0,0,1,0,init_dip_t])]
state_history_meas = [array([0,0,1,0,init_dip])]

u = del_x
#R = array([[1,0,0,0,0],[0,pi/100,0,0,0],[0,0,.001,0,0],[0,0,0,1,0],[0,0,0,0,2]])
#R = array([[30*30,0,0],[0,30*30,0],[0,0,30*30]])
R = array([[std([(m-t) for m,t in zip(measured_x,truth_x)])**2,0,0],[0,std([(m-t) for m,t in zip(measured_y,truth_y)])**2,0],[0,0,std([(m-t) for m,t in zip(measured_z,truth_z)])**2]])
print R 
#def make_coords(state,phix,meas):
#	x_coord = meas[0] + state[0]*abs(cos(state[1])) - (a/(state[2]))*(cos(state[1])-cos(state[1]+phix))
#	y_coord = meas[1] + state[0]*abs(sin(state[1])) - (a/(state[2]))*(sin(state[1])-sin(state[1]+phix))
#	z_coord = meas[2] + state[3] + (a/(state[2]))*(state[4])*phix
#	return array([x_coord,y_coord,z_coord])
def make_prediction_matrix(del_phi,dak0,dak1,k,tanl):
	F_matrix = array([ 
	[cos(del_phi),dak1*sin(del_phi),(a/(k*k))*(1-cos(del_phi)),0,0], #d dP/da
	[-(1/dak1)*sin(del_phi),(dak0/dak1)*cos(del_phi),(a/(k*k*(dak1)))*sin(del_phi),0,0], # dphi/da
	[0,0,1,0,0],#dk/da
	[(a/(k*dak1))*tanl*sin(del_phi),(a/k)*tanl*(1-(dak0/dak1)*cos(del_phi)),(a/(k*k))*tanl*(del_phi-(a/(k*dak1))*sin(del_phi)),1,-del_phi*(a/k)], #ddz/da
	[0,0,0,0,1] #dtanl/da
	])
	return F_matrix
def make_projection_matrix(state,phi):
#	H_matrix = array([
##	[cos(state[1]),   -(state[0]+(a/state[2]))*sin(state[1]),   -(a/(state[2]*state[2]))*cos(state[1]), 0  ,0], # dx/da
#	[cos(state[1]),   -0*(state[0]+(a/state[2]))*sin(state[1]),   (a/(state[2]*state[2]))*cos(state[1]), 0  ,0], # dx/da
#	[sin(state[1]),    0*(state[0]+(a/state[2]))*cos(state[1]),   (a/(state[2]*state[2]))*sin(state[1]), 0  ,0], # dy/da
##	[sin(state[1]),    (state[0]+(a/state[2]))*cos(state[1]),   -(a/(state[2]*state[2]))*sin(state[1]), 0  ,0], # dy/da
#	#[0,               0,                                    (a/(state[2]*state[2]))*state[4]*phi, -1, (a/state[2])*phi] #dz/da
#	[0,               0,                                    (a/(state[2]*state[2]))*state[4]*phi, -1, (a/state[2])*state[1]] #dz/da
#	])

	delphi = ang_steps
	#H_matrix = array([
	#[cos(state[1]),   -(state[0]-(a/state[2]))*sin(state[1]) + (a/state[2])*sin(state[1]+delphi),   -(a/(state[2]*state[2]))*(cos(state[1])-cos(state[1]+delphi)), 0  ,0], # dx/da
	#[sin(state[1]),  (state[0]-(a/state[2]))*cos(state[1]) - (a/state[2])*cos(state[1]+delphi),   -(a/(state[2]*state[2]))*(sin(state[1])-sin(state[1]+delphi)), 0  ,0], # dy/da
	#[0,               0,                                    (a/(state[2]*state[2]))*state[4]*phi, -1, (a/state[2])*state[1]] #dz/da
	#])
	H_matrix = array([
	[cos(state[1]),   -(state[0]-(a/state[2]))*sin(state[1]) + (a/state[2])*sin(state[1]+delphi),   -(a/(state[2]*state[2]))*(cos(state[1])-cos(state[1]+delphi)), 0  ,0], # dx/da
	[sin(state[1]),  (state[0]-(a/state[2]))*cos(state[1]) - (a/state[2])*cos(state[1]+delphi),   -(a/(state[2]*state[2]))*(sin(state[1])-sin(state[1]+delphi)), 0  ,0], # dy/da
	[0,               0,                                    (a/(state[2]*state[2]))*state[4]*phi, -1, (a/state[2])*state[1]] #dz/da
	])
	#print H_matrix/
	return H_matrix
def predict_step(x_cur,p_cur,step,del_phi,dak0,dak1,k,tanl):
	F = make_prediction_matrix(del_phi,dak0,dak1,k,tanl)
	B = array([0,0,0,0,0])
	
	suppression = 1000
	Q = suppression*array([
	[0,0,0,0,0],
	#[0,.001*tanl,0,0,0],#[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
	[0,(1+(tanl*tanl)),0,0,0],#[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
	[0,0,(k*tanl)*(k*tanl),0,k*tanl*(1+(tanl*tanl))],#[0,0,0,0,0],[0,0,0,0,0]])
	[0,0,0,0,0],
	[0,0,k*tanl*(1+(tanl*tanl)),0,(1+(tanl*tanl))*(1+(tanl*tanl))]])
	#Q = array([[0,0,0,0,0],[0,10001,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,1]])
	x_pre = dot(F,x_cur) + dot(B,u)
	#x_pre[1] = x_pre[1]%(2*pi)
	print "x_pre: ",x_pre
	p_pre = dot(dot(F,p_cur),transpose(F)) + Q
	#p_pre = array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])#dot(dot(F,p_cur),transpose(F)) + Q
	return x_pre,p_pre

def update_step(step,x_cur,p_cur):
	#x_pre, p_pre = predict_step(x_cur,p_cur,step)
	phi0 = x_cur[1]
	tanl = x_cur[4]
	k = x_cur[2]
	dp0 = x_cur[0]
	dak0 = (dp0+a/k)
	#if measured_x[step]-measured_x[step-1] == 0:
	#	phix = pi/2.
	#else:
	#	phix= atan((measured_y[step]-measured_y[step-1])/(measured_x[step]-measured_x[step-1]))
	phix = (2*pi/(2*num_steps))

	#z = array([measured_x[step],measured_y[step],measured_z[step]])
	z0 = array([measured_x[step-1],measured_y[step-1],measured_z[step-1]])
	z1 = array([measured_x[step],measured_y[step],measured_z[step]])
	#print "meas: "
	zp = convert_meas(z0,z1,state_history_meas[-1],True,step)	
	z= z1
	#t0 = array([truth_x[step-1],truth_y[step-1],truth_z[step-1]])
	#t1 = array([truth_x[step],truth_y[step],truth_z[step]])
	#print "true: "
	#tt = convert_meas(t0,t1,state_history_true[-1],False,step)
#	tt2 = tt
	#print "meas: ",z
	#print "truth: ",tt
	#H = array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
	#H = make_projection_matrix(x_cur,phix)
	z_state =zp# dot(transpose(H),z)
	#print "z state: ", z_state
	dp1 = z_state[0]
	phi1 = z_state[1]
	dak1 = (dp1+a/k)
	#del_phi = phi1-phi0
	del_phi = ang_steps 
	#if (phi1 <0 and phi0 > 0): 
#		del_phi = phi1+pi - phi0
#	elif (phi1>0 and phi0<0):
#		del_phi = phi1-pi - phi0
#	print "del_phi: ", del_phi	
	x_pre, p_pre = predict_step(x_cur,p_cur,step, del_phi,dak0,dak1,k,tanl)#(10./pi))#tanl)
	#x_pre[1] = tt[1]
	H = make_projection_matrix(zp,phix)
	#H = make_projection_matrix(x_cur,phix)
	
	#print dot(transpose(H),z)
	print "Hxpre: ", dot(H,x_pre)
	K = dot(dot(p_pre,transpose(H)), linalg.inv(dot(dot(H,p_pre),transpose(H)) + R))
	#print "K: ",K
	x_new = x_pre + dot(K,(z- dot(H,x_pre)))
	#x_new[1] = x_new[1]%(2*pi)
	#xx = make_coords(x_pre,del_phi,z0,True,state_history[-1][3])
	#x_new = x_pre + dot(K,(z- xx))
	p_new = p_pre - dot(dot(K,H),p_pre)
	#del_phi_t = tt[1]-state_history_true[-1][1]
	del_phi_z = z[1]-state_history_meas[-1][1]
	#tt_coords = make_coords(tt,del_phi_t,state_history_coords_true[-1])
	z_coords = make_coords(zp,del_phi_z,z0,True,state_history_meas[-1][3])

	#t_phi0 = state_history_true[-1][1]
	#t_tanl = state_history_true[-1][4]
	#t_k = state_history_true[-1][2]
	#t_dp0 = state_history_true[-1][0]
	#t_dak0 = (t_dp0+a/t_k)
	#t_dp1 = tt[0]
	#t_phi1 = tt[1]
	#t_dak1 = (t_dp1+a/t_k)
	#t_del_phi = t_phi1-t_phi0
	#FF = make_prediction_matrix(t_del_phi,t_dak0,t_dak1,t_k,t_tanl)
	#print "tt step %s: "%(step), tt
#	print "F*tt: ", dot(FF,tt)
#	print "Cos: ",cos(t_del_phi), t_dp1*cos(t_del_phi)
#	print "sin: ",t_dak0*sin(t_del_phi), t_phi1*t_dak0*sin(t_del_phi)
#	print "k: ",(1*a/(t_k*t_k))*(1-cos(t_del_phi))
#	print "diff: ",dot(array([cos(t_del_phi),0*t_dak0*sin(t_del_phi),(1*a/(t_k*t_k))*(1-cos(t_del_phi)),0,0]),tt)

#	k_coords = make_coords(x_new,del_phi,state_history_coords[-1],True,state_history[-1][3])
#	k_coords = make_coords(x_new,del_phi,z0,True,state_history[-1][3])
	k_coords = dot(H,x_new)
	
	#print "truth act: ",t1
	#print "truth coords: ",tt_coords
	#print "truth param: ",tt
	#print "meas act: ",z1
	#print "meas coords: ",z_coords
	#print "meas param: ",zp
	#print "kal param: ", x_new
	#print "kal: ", k_coords #make_coords(x_new,del_phi,z,True,state_history_coords[step-1][3]))
	state_history.append(x_new)
	state_history_coords.append(k_coords)#state_history_coords[step-1]))
	#state_history_coords.append(make_coords(x_new,del_phi,z0))#state_history_coords[step-1]))
	#state_history_coords.append(make_coords(x_new,del_phi,state_history_coords[-1]))
	#state_history_true.append(tt)
	state_history_meas.append(zp)
	#state_history_coords_true.append(tt_coords)
	state_history_coords_meas.append(z_coords)
	return x_new, p_new
for num in range(len(measured_x)-1):
	current_state, current_un = update_step(num+1,current_state,current_un)
kal_x = []
kal_y = []
kal_z = []
for i,state in enumerate(state_history_coords):
	kal_x.append(state[0])
	kal_y.append(state[1])
	kal_z.append(state[2])
	#if i < 100:
	#	print state_history[i]
#	print "true: ",state_history_true[i]
#	print "kal: ",state_history[i]
##############################################################
fig=plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.plot(kal_x,kal_y,kal_z,c='g')
ax.plot(truth_x,truth_y,truth_z,c='r')
ax.plot(measured_x,measured_y,measured_z,c='b')
plt.show()

ax = plt.subplot(111)
ax.set_xlabel("step")
ax.set_ylabel("dis")
ax.plot(range(len(truth_x)), [sqrt( (kx-tx)**2 + (ky-ty)**2 + (kz-tz)**2) for kx,ky,kz,tx,ty,tz in zip(kal_x,kal_y,kal_z,truth_x,truth_y,truth_z)],'g')
ax.plot(range(len(truth_x)), [sqrt( (mx-tx)**2 + (my-ty)**2 + (mz-tz)**2) for mx,my,mz,tx,ty,tz in zip(measured_x,measured_y,measured_z,truth_x,truth_y,truth_z)],'r')
print "kal: ",  average([sqrt( (kx-tx)**2 + (ky-ty)**2 + (kz-tz)**2) for kx,ky,kz,tx,ty,tz in zip(kal_x,kal_y,kal_z,truth_x,truth_y,truth_z)])
print "meas: ", average([sqrt( (mx-tx)**2 + (my-ty)**2 + (mz-tz)**2) for mx,my,mz,tx,ty,tz in zip(measured_x,measured_y,measured_z,truth_x,truth_y,truth_z)])
#ax.plot(range(len(truth_x)), [sqrt( (kx-tx)**2 + (ky-ty)**2) for kx,ky,kz,tx,ty,tz in zip(kal_x,kal_y,kal_z,truth_x,truth_y,truth_z)],'g')
#ax.plot(range(len(truth_x)), [sqrt( (mx-tx)**2 + (my-ty)**2) for mx,my,mz,tx,ty,tz in zip(measured_x,measured_y,measured_z,truth_x,truth_y,truth_z)],'r')
#print "kal: ",  average([sqrt( (kx-tx)**2 + (ky-ty)**2) for kx,ky,kz,tx,ty,tz in zip(kal_x,kal_y,kal_z,truth_x,truth_y,truth_z)])
#print "meas: ", average([sqrt( (mx-tx)**2 + (my-ty)**2) for mx,my,mz,tx,ty,tz in zip(measured_x,measured_y,measured_z,truth_x,truth_y,truth_z)])
plt.show()

ax = plt.subplot(111)
ax.set_xlabel("step")
ax.set_ylabel("dis")
ax.plot(range(len(truth_x[num_steps/10:])), [sqrt( (kx-tx)**2 + (ky-ty)**2 + (kz-tz)**2) for kx,ky,kz,tx,ty,tz in zip(kal_x[num_steps/10:],kal_y[num_steps/10:],kal_z[num_steps/10:],truth_x[num_steps/10:],truth_y[num_steps/10:],truth_z[num_steps/10:])],'g')
ax.plot(range(len(truth_x[num_steps/10:])), [sqrt( (mx-tx)**2 + (my-ty)**2 + (mz-tz)**2) for mx,my,mz,tx,ty,tz in zip(measured_x[num_steps/10:],measured_y[num_steps/10:],measured_z[num_steps/10:],truth_x[num_steps/10:],truth_y[num_steps/10:],truth_z[num_steps/10:])],'r')

print "kal cut: ",  average([sqrt( (kx-tx)**2 + (ky-ty)**2 + (kz-tz)**2) for kx,ky,kz,tx,ty,tz in zip(kal_x[num_steps/10:],kal_y[num_steps/10:],kal_z[num_steps/10:],truth_x[num_steps/10:],truth_y[num_steps/10:],truth_z[num_steps/10:])])
print "meas cut: ", average([sqrt( (mx-tx)**2 + (my-ty)**2 + (mz-tz)**2) for mx,my,mz,tx,ty,tz in zip(measured_x[num_steps/10:],measured_y[num_steps/10:],measured_z[num_steps/10:],truth_x[num_steps/10:],truth_y[num_steps/10:],truth_z[num_steps/10:])])
plt.show()

ax = plt.subplot(111)
ax.set_xlabel("step")
ax.set_ylabel("dis")
ax.plot(range(len(truth_x)), [(kx-tx) for tx,kx in zip(truth_x,kal_x)],'g')
ax.plot(range(len(truth_x)), [(ky-ty) for ty,ky in zip(truth_y,kal_y)],'r')
#ax.plot(range(len(truth_x)), [(kz-tz) for tz,kz in zip(truth_z,kal_z)],'g')
#ax.plot(range(len(truth_x)), [(mz-tz) for tz,mz in zip(truth_z,measured_z)],'r')
plt.show()

ax = plt.subplot(111)
ax.set_xlabel("step")
ax.set_ylabel("x")
ax.plot(range(len(truth_x)), [kx for kx in kal_x],'g')
ax.plot(range(len(truth_x)), [tx for tx in truth_x],'b')
ax.plot(range(len(truth_x)), [ky for ky in kal_y],'m')
ax.plot(range(len(truth_x)), [ty for ty in truth_y],'r')
#ax.plot(range(len(truth_x)), kal_z,'g')
#ax.plot(range(len(truth_x)), measured_z,'r')
#ax.plot(range(len(truth_x)), truth_z,'b')
plt.show()
ax = plt.subplot(111)
ax.set_xlabel("step")
ax.set_ylabel("angle")
ax.plot(range(len(truth_x)), [kx[1] for kx in state_history],'g')
#ax.plot(range(len(truth_x)), [tx[1] for tx in state_history_true],'b')
ax.plot(range(len(truth_x)), [mx[1] for mx in state_history_meas],'r')
plt.show()
