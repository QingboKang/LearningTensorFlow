import tensorflow as tf

a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')

x = tf.add(a, b, name='add')
y = tf.multiply(a, b, name='mul')

sess = tf.Session()
x, y = sess.run([x, y])
print(x)
print(y)

print(sess.graph.as_graph_def())

writer = tf.summary.FileWriter("H:/tflog", tf.get_default_graph())
writer.close()