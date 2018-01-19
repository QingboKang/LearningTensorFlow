import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.constant([3, 3, 3], dtype=tf.float32)

c = a + b

sess = tf.Session()

print(sess.run(c, {a: [1, 2, 34]}))



writer = tf.summary.FileWriter("H:/tflog", tf.get_default_graph())
writer.close()