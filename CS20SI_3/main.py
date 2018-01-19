import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

import utils

DATA_FILE = "slr05.xls"

# Step 1: read in data from .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X and label Y
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Step 3: create weight and bias
w = tf.Variable(0.0, name="weights_1")
u = tf.Variable(0.0, name="weights_2")
b = tf.Variable(0.0, name ="bias")

# Step 4: construct model to predict Y from X
Y_pred = X * X * w + X * u + b

# Step 5: use the square error as the loss function
# loss = tf.square(Y-Y_pred, name="loss")
loss = utils.huber_loss(Y, Y_pred)

# Step 6: using gradient descent with learning rate to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000001).minimize(loss)

sess = tf.Session()
# Step 7: initialize the necessary variables, w and b
sess.run(tf.global_variables_initializer())

num_epoches = 100
every_loss = np.zeros( num_epoches)
# Step 8: train the model
for i in range(num_epoches):   # run 100 epochs
    total_loss = 0
    for x, y in data:
        _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
        total_loss += l
    every_loss[i] = total_loss/n_samples
    print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

# Step 9: output the values of w and b
w_value, u_value, b_value = sess.run([w, u, b])

writer = tf.summary.FileWriter("H:/tflog", tf.get_default_graph())

writer.close()

print("Weight: ", w_value, u_value)
print("b: ", b_value)

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X,  X * X * w_value + X * u_value + b_value, 'ro', label='Predicted data')
plt.legend()
plt.show()

# plot the loss for every iteration
plt.figure(2)
plt.plot(np.linspace(1, num_epoches, num_epoches), every_loss, 'r')
plt.show()