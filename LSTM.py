import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 입력 시퀀스 데이터 생성
X = np.array([[[1], [2], [3], [4], [5]],
              [[6], [7], [8], [9], [10]],
              [[11], [12], [13], [14], [15]]])

# 출력 시퀀스 데이터 생성
y = np.array([[6], [11], [16]])

# RNN 모델 생성
model = Sequential()
model.add(LSTM(10, input_shape=(5, 1)))  # 입력 차원: (시퀀스 길이, 특성 수)
model.add(Dense(1))

# 모델 컴파일
model.compile(loss='mean_squared_error', optimizer='adam')

# 모델 학습
model.fit(X, y, epochs=10, batch_size=1, verbose=2)

# 테스트 데이터 생성
X_test = np.array([[[16], [17], [18], [19], [20]]])

# 예측
y_pred = model.predict(X_test)
print(y_pred)
