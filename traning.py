import imageread as image
import tensorflow as tf
import model

x_data, y_data = image.readimages('img/')

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    model.loadLearning(sess)

    for i in range(1000):
        sess.run(model.train_op, feed_dict={model.X: x_data, model.Y: y_data, model.p_keep_conv: 0.8, model.p_keep_hidden: 0.5})
        print i, sess.run(model.cost, feed_dict={model.X: x_data, model.Y: y_data, model.p_keep_conv: 0.8, model.p_keep_hidden: 0.5})

    model.saveLearning(sess)
