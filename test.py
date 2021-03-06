import cv2
import tensorflow as tf

import model

face_cascade = cv2.CascadeClassifier('frontalFace/haarcascade_frontalface_alt.xml')
cap= cv2.VideoCapture(0)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    model.loadLearning(sess)

    while(True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)


        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = img[y:y+h, x:x+w]
            data = cv2.resize(face,(28,28))

            num = sess.run(model.predict_op, feed_dict={model.X: [data], model.p_keep_conv: 1, model.p_keep_hidden: 1})
            string = "picture :" + str(num)
            cv2.putText(img, string, (x,y+h+50), cv2.FONT_HERSHEY_SIMPLEX, 2, 200)


        cv2.imshow('face detaction test', img)
        cv2.waitKey(12)
