# tensorflow-cnn-face_analysis

기본적인 cnn과 opencv를 활용하여 만들고 있는 얼굴 분석 모듈 입니다<br>
카메라로 얼굴을 추출하고 그 얼굴이 누구인지 알려줍니다.<br>

---

# How to use

traning.py는 모델을 트레이닝 하는 코드이며<br>
test.py는 학습된 모델을 불러와 opencv로 cam 프레임을 받아와 분석하는 코드입니다.

```
$ python traning.py
$ python test.py
```
---

# Test your own picture

img디렉터리에 당신이 가지고 있는 이미지를 넣어 테스트를 해볼 수 있습니다.<br>

### 소스를 수정하지 않고 테스트 하는 방법

각각의 이미지 디렉터리 1, 2, 3, 4에 자신이 원하는 이미지를 넣고<br>
cps 디렉터리안의 파일을 전부 삭제 해 주시고 실행 해 주세요   

### 소스를 수정하여 테스트 하는 방법

당신이 분류할 사진 수를 늘리거나 줄일 경우 수정해야할 소스는<br>
test.py, imageread.py, model.py 총 3개 입니다.<br>

#### test.py

```
if(sess.run(model.predict_op, feed_dict={model.X: [data], model.p_keep_conv: 1, model.p_keep_hidden: 1})[0]) == 0:
    print "picture 1"

elif(sess.run(model.predict_op, feed_dict={model.X: [data], model.p_keep_conv: 1, model.p_keep_hidden: 1})[0]) == 1:
    print "picture 2"

elif (sess.run(model.predict_op, feed_dict={model.X: [data], model.p_keep_conv: 1 , model.p_keep_hidden: 1})[0]) == 2:
    print "picture 3"

elif (sess.run(model.predict_op, feed_dict={model.X: [data], model.p_keep_conv: 1, model.p_keep_hidden: 1})[0]) == 3:
    print "picture 4"
```

#### imageread.py
```
y_dataset = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
```

#### model.py
```
Y = tf.placeholder("float", [None, 4])
w_o = init_weights([625, 4])  
```

이 부분들의 코드를 자신의 사진 종류 수에 맞게 변형하세요.<br>
그 다음 cps 디렉터리 안의 파일을 전부 삭제 해 주시고 실행 해 주세요<br>
제가 표시한 부분 말고도 수정이 가능한 부분은 많습니다. 자신의 기호에 따라 **변형** 하세요.<br>


---
# 허접으로 개발을 해보며
제 소스코드는 좋은편이 아닙니다. 인스턴스하게 기본적인 cnn코드를 활용한 것으로<br>
이미지 전처리, cnn구조등 여러가지 개선사항이 눈에 보입니다.<br><br>
또한 데이터 양의 부족, 데이터에 대한 이해 부족 등으로 인하여 인식률은 그리 높지 않습니다.<br>
**데이터 이해 못하면 머신러닝은 돌아가지 않는다는 것을 깨달았습니다.**
