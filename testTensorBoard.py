import tensorflow as tf

a = tf.constant(2.0, name = "a")
b = tf.constant(3.0, name = "b")

x = tf.add(a, b, name = "add")

sess = tf.Session()

print(sess.run(x))

writer = tf.summary.FileWriter("H:/tflog", tf.get_default_graph())

writer.close()