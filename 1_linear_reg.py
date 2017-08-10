import tensorflow as tf
import numpy as np

# create some data
X = np.asarray(np.random.normal(size = (10000,9)))
weights = np.asarray(np.random.randint(low=6, size=(9,1))) + np.asarray(np.random.normal(size=(9,1)))
bias = np.asarray(np.random.randint(low=9, size=(1,1))) + np.asarray(np.random.normal(size=(1,1)))
Y = np.dot(X,weights)+bias

# define paramaters
train_fraction = 0.7 
batch_size = 100
n_output = 1 # the number of output variables
n_epochs = 100 # number of epochs
learning_rate = 0.001 
display_rate = 5 # test set cost will be displayed after display_rate number of epochs

# split into test and train
train_size = int(X.shape[0]*train_fraction)
test_size = X.shape[0] - train_size
train_ind = np.full(shape=(X.shape[0],), fill_value = False)
train_ind[np.random.choice(np.arange(0,X.shape[0]), size = train_size, replace = False)] = True

train_X, test_X = X[train_ind,:], X[np.logical_not(train_ind),:]
train_Y, test_Y = Y[train_ind,:], Y[np.logical_not(train_ind),:]

batch_ind = list(set(list(range(0,train_size, batch_size))+ [train_size-1]))
batch_ind.sort()
num_batches = len(batch_ind)-1

n_features = X.shape[1]
costs = []

x_train_batch = tf.placeholder('float64', (None,n_features), name = 'x')
y_train_batch = tf.placeholder('float64', (None,n_output), name = 'y')

w = tf.Variable(np.asarray(np.random.normal(size=(n_features,n_output))),trainable=True, name = 'train_weights', dtype = 'float64')
b = tf.Variable(np.asarray([0]),trainable=True, name = 'train_bias', dtype = 'float64')

y_hat = tf.add(tf.matmul(x_train_batch,w), b)
cost = tf.reduce_mean(tf.square(y_train_batch-y_hat))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

_ = tf.global_variables_initializer()

sess = tf.Session()
sess.run(_)

costs_train = []
costs_test = []

for epoch in range(n_epochs):
    for batch_pos in range(num_batches):
        x_batch = train_X[batch_ind[batch_pos]:batch_ind[batch_pos+1]]
        y_batch = train_Y[batch_ind[batch_pos]:batch_ind[batch_pos+1]]
        sess.run(optimizer, feed_dict = {x_train_batch: x_batch,\
                 y_train_batch: y_batch})
        
    cost_train = sess.run(cost, feed_dict = {x_train_batch: train_X,\
             y_train_batch: train_Y})
    cost_test = sess.run(cost, feed_dict = {x_train_batch: test_X,\
             y_train_batch: test_Y})
    costs_train.append(cost_train)
    costs_test.append(cost_test)
    
    if (epoch)%display_rate == 0:
        cost_ = sess.run(cost, feed_dict = {x_train_batch: train_X,\
                 y_train_batch: train_Y})
        print('Epoch:', epoch, 'Cost: ', cost_)

# save the model
sess.run(w)
saver = tf.train.Saver() 
saver.save(sess, 'linear_reg.chkp')

# plot the training
from matplotlib import pyplot as plt
plt.plot(costs_train, label='Change in training costs.')
plt.plot(costs_test, label='Change in testing costs.')
plt.show()


# restore the session
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'linear_reg.chkp')

