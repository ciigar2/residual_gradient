import tensorflow as tf
import numpy, sys
import gym

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def get_next_episode_data(env):
	s_li=[]
	s_prime_li=[]
	r_li=[]
	s=env.reset()
	while(True):
		s_li.append(s)
		if s[1]>0:
			a=2
		else:
			a=0
		s,r,done,_=env.step(a)
		s_prime_li.append(s)
		r_li.append(r)
		if done==True:
			break
	s_prime_li[-1]=numpy.array([0,0])
	T=len(s_li)
	state_size=len(s)
	#print(s_li)
	#sys.exit(1)
	return numpy.array(s_li).reshape(T,state_size),numpy.array(s_prime_li).reshape(T,state_size),numpy.array(r_li).reshape(T,1)

class q_network:
	def __init__(self,params):
		self.state_size=params['state_size']
		self.output_size=params['output_size']
		self.width=params['width']
		self.alpha=params['alpha']
		self.gamma=0.99
		
		
		#define net weights
		self.weights=[tf.Variable(tf.random_normal([self.width, self.width], stddev=0.35),
                      name="weights") for _ in range(params['num_hidden_layers']-1)]
		self.weights.insert(0,tf.Variable(tf.random_normal([self.state_size, self.width], stddev=0.35)))
		self.weights.append(tf.Variable(tf.random_normal([self.width, 1], stddev=0.35)))
		#define net weights




		self.biases=[tf.Variable(tf.ones([self.width])) for _ in range(params['num_hidden_layers'])]
		self.biases.append(tf.Variable(tf.ones([1])))

		self.network()
		return

	def network(self):
		self.state = tf.placeholder(tf.float32, [None,self.state_size], "state")
		self.next_state = tf.placeholder(tf.float32,[None,self.state_size],"next_state")
		self.reward=tf.placeholder(tf.float32,[None,1],"rewards")
		self.non_terminal=tf.placeholder(tf.float32,[None,1],"nonTerminal")

		self.value=self.state
		for h in range(params['num_hidden_layers']):
			self.value = tf.nn.relu(tf.add(tf.matmul(self.value,self.weights[h]),self.biases[h]))
		self.value = tf.matmul(self.value,self.weights[-1])+self.biases[-1]
		
		self.next_value=self.next_state
		for h in range(params['num_hidden_layers']):
			self.next_value = tf.nn.relu(tf.matmul(self.next_value,self.weights[h])+self.biases[h])
		self.next_value = tf.matmul(self.next_value,self.weights[-1])+self.biases[-1]
		# i want s[terminal] to be zero, but it is not zero in this case ...!
		self.loss_vec=self.reward+tf.multiply(self.non_terminal,self.gamma*tf.stop_gradient(self.next_value))-self.value
		#self.loss_vec=self.reward+tf.multiply(self.non_terminal,self.gamma*self.next_value)-self.value
		self.loss = tf.reduce_mean(tf.square(self.loss_vec))

		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.loss)


params={}
params['state_size']=2
params['output_size']=1
params['width']=60
params['alpha']=0.0005
params['num_hidden_layers']=3

tf.reset_default_graph()
with tf.Session() as sess:
	nn=q_network(params)
	init=tf.initialize_all_variables()
	sess.run(init)
	env=gym.make('MountainCar-v0')

	for episodes in range(1000):
		s,s_prime,r=get_next_episode_data(env)
		non_terminal=numpy.ones_like(r)
		non_terminal[-1]=0
		_,l,loss_vec=sess.run([nn.optimizer, nn.loss,nn.loss_vec],feed_dict={nn.state:s,
											nn.next_state:s_prime,
											nn.reward:r,
											nn.non_terminal:non_terminal}
					)

		if episodes%50==0:
			print(l)
			#print(sess.run(nn.value,feed_dict={nn.state:[numpy.array([0,0])]}))
		'''
		if episodes==500:
			print(loss_vec)
			sys.exit(1)
		'''

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	# Make data.
	X = numpy.arange(-1, .6, 0.005)
	Y = numpy.arange(-0.06, 0.06, 0.005)
	Z=numpy.zeros((len(X),len(Y)))



	X1, Y1 = numpy.meshgrid(X, Y)
	R = numpy.sqrt(X1**2 + Y1**2)
	Z = numpy.sin(R)
	#sys.exit(1)
	for i,x in enumerate(X):
		for j,y in enumerate(Y):
			Z[j,i]=(sess.run(nn.value,feed_dict={nn.state:[numpy.array([x,y])]}))[0]

	# Plot the surface.
	surf = ax.plot_surface(X1, Y1, Z, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)

	# Customize the z axis.
	#ax.set_zlim(-1.01, 1.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()



