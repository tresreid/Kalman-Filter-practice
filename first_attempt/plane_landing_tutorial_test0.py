## home..wlu.edu/~levys/kalman_tutorial
## Sample Kalman filter in 1-D using only a scalar case. 

from random import *
import matplotlib.pyplot as plt
import numpy as np



## altitude_current = (percent lower) * altitude_previous + turbulence
## => x_k = a*x_k-1 + w_k
## observed_altitude_current = altitude_current + noise_current
## => z_k = x_k + v_k ==> x_k = z_k - v_k (however, we do not know v_k directly, just estimate it via a kalman gain
## kalman gain g ~> x^_k = x^_k-1 + g_k(z_k - x^_k-1)

# make from gain
if False:
	x1 =110
	z1 = 105
	gk = [.1*x for x in range(10)]
	x = [x1 +g*(z1-x1) for g in gk]
	print x
# compute gain
# we can compute the gain indirectly from the noise average r. 
# => g_k = p_k-1 / (p_k-1 +r) where p_k is the prediction error computed recursively. p_k = (1-g_k)p_k-1
# the constant is applied both to the prediction and the prediction error
# x_k = a*x_k-1  p_k = a*p_k-1 *a

# predict step
# x_k = a_k-1
# p_k = a * p_k-1 *a

# update step
# g_k = p_k / (p_k +r)
# x_k <- x_k + g_k(z_k-x_k)
# p_k <- (1-g_k)p_k

###############################
## test 1
############################3##

real_data = [1000]
for i in range(10):
	real_data.append(real_data[i]*.75)
#measured_data = [ real + randint(-200,200) for real in real_data]
measured_data = [805,920,580,481,210,247,19,54,174,211,19]
print real_data
print measured_data

predicted_data = [measured_data[0]]
prediction_err = [1]
kalman_gain = [-1]
def predict(x_cur,p_cur):
	a = 0.75
	x_pre = a* x_cur
	p_pre = a* p_cur * a
	return (x_pre,p_pre)
def update(step):
	r = 100
	x_pre,p_pre = predict(predicted_data[step-1],prediction_err[step-1])
	up_gain =(p_pre/(p_pre+r))
	up_data = x_pre+ up_gain*(measured_data[step]-x_pre)
	up_err = (1-up_gain)*p_pre
	return up_data,up_err,up_gain
for s in range(10):
	data,err, gain =update(s+1)
	predicted_data.append(data)
	prediction_err.append(err)
	kalman_gain.append(gain)

print predicted_data
print prediction_err
print kalman_gain

ax = plt.subplot(111)
plt.plot(range(11), real_data, "b-")
plt.plot(range(11), measured_data, "r-")
plt.plot(range(11), predicted_data, "g-")
plt.show()
