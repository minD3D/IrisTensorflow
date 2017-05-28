from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# 데이터셋
IRIS_TRAINING = "train.csv"
IRIS_TEST = "test.csv"

# 데이터셋을 불러옵니다.
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.int)

# 모든 특성이 실수값을 가지고 있다고 지정합니다
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=5)]


# 10, 20, 10개의 유닛을 가진 3층 DNN를 만듭니다
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[3, 3, 5],
                                            n_classes=5,
                                            model_dir="/tmp/iris_model")

# 모델을 학습시킵니다.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=6000)

# 정확도를 평가합니다.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('정확도: {0:f}'.format(accuracy_score))
