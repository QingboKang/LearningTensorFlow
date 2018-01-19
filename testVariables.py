import tensorflow as tf

# create variable a with scalar value
a = tf.Variable(2, name="scalar")

# create variable b as a vector
b = tf.Variable([3, 4], name="vector")

# create variable c as a 2 * 2 matrix
c = tf.Variable([[0, 1], [3, 4]], name="matrix")

# create variable w as 100*10 tensor, filled with zeros
w = tf.Variable(tf.zeros([100, 10]))

# initialize variables
init = tf.global_variables_initializer()
init_ab = tf.variables_initializer([a, b], name="init_ab")
sess = tf.Session()
sess.run(init_ab)

#
W_rand = tf.Variable(tf.truncated_normal([2, 3]))
sess.run(W_rand.initializer)

print(W_rand.eval(sess))

W = tf.Variable(10)
assign_op = W.assign(100)

sess.run(assign_op)
print(W.eval(sess))

writer = tf.summary.FileWriter("H:/tflog", tf.get_default_graph())
writer.close()