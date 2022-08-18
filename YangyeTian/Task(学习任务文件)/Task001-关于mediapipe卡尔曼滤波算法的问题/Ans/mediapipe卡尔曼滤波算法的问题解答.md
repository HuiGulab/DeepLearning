# 关于mediapipe卡尔曼滤波算法的问题

## 1.请描述卡尔曼滤波算法的基本原理

**滤波**：使受噪声影响的数据更加趋于真实值。

**使用系统**：线性高斯系统

​	**线性**：叠加性：$F(x_1 + x_2) = F(x_1)+F(x_2)$

​				齐次性：$k \times  F(x) = F(k \times x)$

​	**高斯**：噪声满足正太分布



### 状态空间表达式

- **状态方程**：$X_k = A \times X_{k-1} + B \times u_k + w_k$

  其中$X_k$表示当前状态值，$X_{k-1}$表示上一个时刻输入的状态值，$u_k$表示当前的输入值，$w_k$表示过程噪声,$A$表示状态转移矩阵,$B$表示控制转移矩阵。

- **观测方程**：$Y_k = C \times X_k + v_k$

  $X_k$乘上某种关系矩阵$C$得到对应的观测值$Y_k$，$v_k$表示观测噪声。

![image-20220817145932368](.\mediapipe卡尔曼滤波算法的问题解答.assets\image-20220817145932368.png)

两噪音符合高斯分布$w_k ∈ N(0;Q_k)$、$v_k ∈ N(0;R_k)$

### 具体公式

**卡尔曼滤波**就是对k时刻的状态值进行一个预估值X，和对该时刻一个测量值Y，二者都有误差，最后卡尔曼滤波使用二者的加权值作为该时刻的最终预测值。

 ![image-20220817155620871](.\mediapipe卡尔曼滤波算法的问题解答.assets\image-20220817155620871.png)

对于一维卡尔曼来说，卡尔曼权重$K_t = \frac{P_{t-1} + Q }{P_{t-1} + Q + R} $，预估值噪音方差Q越大，K值就越大，相当于更相信测量值；测量值噪音方差R越大，K就越小，相当于越相信预估值。

## 2.附件代码已实现卡尔曼滤波算法对单个2D关键点的实时平滑处理

**运行效果中红色为噪音，绿色为最佳预测点**

```python
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# 使用手部模型
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
# mpHolistic = mp.solutions.holistic
# holistic = mpHolistic.Holistic()


# 定义x的初始状态 -- 需要修改，初始化为捕获手势的位置x坐标
x_mat = np.mat([[0, ], [0, ], [0, ], [0, ]])
# 定义初始状态协方差矩阵
p_mat = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# 定义初始化控制矩阵
b_mat = np.mat([[0.5, ], [1, ], [0.5, ], [1, ]])
# 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
f_mat = np.mat([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
# 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
q_mat = np.mat([[0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 0.3]])
# 定义观测矩阵
h_mat = np.mat([1, 0, 1, 0])
# 定义观测噪声协方差
r_mat = np.mat([1])

# 获取摄像头编号，笔记本摄像头默认为0
video = cv2.VideoCapture(0)
first_frame_flag = True
while True:
    ret, frame = video.read()
    # opencv读出来的图片为gbr图片，使用cvtColor转变为rgb图片
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 结果侦测
    results = hands.process(img)
    # results = holistic.process(img)

    h, w, c = img.shape
    # multi_hand_landmarks会回传手部21个点的坐标
    if results.multi_hand_landmarks:
        # 如果侦测到手，则通过for循环，将每只手都标出来
        for handLms in results.multi_hand_landmarks:
            # 将手21个点遍历
            for id, lm in enumerate(handLms.landmark):
                # 计算回实际像素点位置
                cx, cy = int(lm.x * w), int(lm.y * h)
                # 只选择第9个点标出
                if id == 9:
                    # 第一次检测
                    if first_frame_flag:
                        x_mat[0, 0] = cx
                        x_mat[2, 0] = cy
                        first_frame_flag = False
                    else:
                        x_predict = f_mat * x_mat
                        p_predict = f_mat * p_mat * f_mat.T + q_mat
                        x_mat_before = x_mat
                        kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
                        # 出现问题
                        x_mat = x_predict + np.multiply(kalman, (np.mat(
                            [[cx, ], [cx - x_mat_before[0, 0], ], [cy, ], [cy - x_mat_before[2, 0], ]]) - x_predict))
                        p_mat = (np.eye(x_mat.shape[0]) - kalman * h_mat) * p_predict
                        noise_x = cx + int(np.random.normal(0, 1) * 10)
                        noise_y = cy + int(np.random.normal(0, 1) * 10)

                        # 将点绘制上去  BGR样式
                        # 第五个参数表示线样式为空
                        cv2.circle(frame, (noise_x, noise_y), 6, (0, 0, 255), cv2.FILLED)  # 红点
                        cv2.circle(frame, (int(x_mat[0, 0]), int(x_mat[2, 0])), 3, (0, 255, 0), cv2.FILLED)  # 绿点
                    break
                    
    cv2.imshow('video', frame)

    # 执行退出
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
video.release()
```

![image-20220818113154622](.\mediapipe卡尔曼滤波算法的问题解答.assets\image-20220818113154622.png)

## 3.请将卡尔曼滤波算法扩展到mediapipe全身2D关键点并总结实验效果

```python
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# 使用人体模型模型
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   model_complexity=1,
                   smooth_landmarks=True,
                   enable_segmentation=False,
                   smooth_segmentation=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# 定义x的初始状态 -- 需要修改，初始化为捕获手势的位置x坐标
# 为每个点创建矩阵
x = np.zeros((33, 4))
x_mat = np.mat([[0, ], [0, ], [0, ], [0, ]])
# 定义初始状态协方差矩阵
p_mat = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# 定义初始化控制矩阵
b_mat = np.mat([[0.5, ], [1, ], [0.5, ], [1, ]])
# 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
f_mat = np.mat([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
# 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
q_mat = np.mat([[0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 0.3]])
# 定义观测矩阵
h_mat = np.mat([1, 0, 1, 0])
# 定义观测噪声协方差
r_mat = np.mat([1])


# 获取摄像头编号，笔记本摄像头默认为0
video = cv2.VideoCapture(0)
first_frame_flag = np.ones(33)
while True:
    ret, frame = video.read()
    # opencv读出来的图片为gbr图片，使用cvtColor转变为rgb图片
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 结果侦测
    results = pose.process(img)

    h, w, c = img.shape
    # pose_landmarks返回人体姿态的33个点
    if results.pose_landmarks:
        # # 将点画在图像中
        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # # 如果侦测到人体，则通过for循环，遍历每个人
        # for poseLms in results.pose_landmarks.landmark:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            # 计算每个实际像素点位置（x,y）
            cx, cy = int(lm.x * w), int(lm.y * h)
            # 第一次检测
            if first_frame_flag[id]:
                x[id][0] = cx
                x[id][2] = cy
                # 第一次检测最后一个点时，该检测结束
                # if id == 32:
                first_frame_flag[id] = 0
            else:
                # 读取x_(k-1)
                x_mat[0, 0] = x[id][0]
                x_mat[2, 0] = x[id][2]

                x_predict = f_mat * x_mat  # x预估值
                p_predict = f_mat * p_mat * f_mat.T + q_mat  # x方差
                x_mat_before = x_mat  # 记录x_(k-1)

                # cy,cx是观测值
                kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)  # 计算卡尔曼权值
                # 出现问题
                x_mat = x_predict + np.multiply(kalman, (np.mat(
                    [[cx, ], [cx - x_mat_before[0, 0], ], [cy, ], [cy - x_mat_before[2, 0], ]]) - x_predict))
                # 将最佳预测写回x保存
                x[id][0] = x_mat[0, 0]
                x[id][2] = x_mat[2, 0]

                p_mat = (np.eye(x_mat.shape[0]) - kalman * h_mat) * p_predict
                noise_x = cx + int(np.random.normal(0, 1) * 10)
                noise_y = cy + int(np.random.normal(0, 1) * 10)

                # 将点绘制上去  BGR样式
                # 第五个参数表示线样式为空
                # 画圆
                cv2.circle(frame, (noise_x, noise_y), 6, (0, 0, 255), cv2.FILLED)  # 红点
                cv2.circle(frame, (int(x_mat[0, 0]), int(x_mat[2, 0])), 3, (0, 255, 0), cv2.FILLED)  # 绿点
            # break



    cv2.imshow('video', frame)

    # 执行退出
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
video.release()

```

![image-20220818113435198](.\mediapipe卡尔曼滤波算法的问题解答.assets\image-20220818113435198.png)

## 4.请验证mediapipe自带的伪3D关键点检测与2D关键点检测的x，y坐标是否一致？

答：一致；(x,y)分别为宽和高所占的比率，可以通过乘上图片宽高变化为2D(x,y)坐标，z为深度值。

```python
cx, cy = int(lm.x * w), int(lm.y * h)
```

```json
landmark {
  x: 0.9534852504730225
  y: 0.9311989545822144
  z: -0.13945917785167694
}
```

## 5.尝试将卡尔曼滤波算法扩展到mediapipe伪3D关键点的平滑处理中，并总结实验效果

```python
 	import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


class Kalman:
    # point_num:关键点的数量
    def __init__(self, point_num):
        # 定义预测变换矩阵
        self.x_mat = np.mat([[0, ], [0, ], [0, ], [0, ]])
        # 定义初始状态协方差矩阵
        self.p_mat = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # 定义初始化控制矩阵
        self.b_mat = np.mat([[0.5, ], [1, ], [0.5, ], [1, ]])
        # 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
        self.f_mat = np.mat([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        # 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
        self.q_mat = np.mat([[0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 0.3]])
        # 定义观测矩阵
        self.h_mat = np.mat([1, 0, 1, 0])
        # 定义观测噪声协方差
        self.r_mat = np.mat([1])
        # 定义存储上一个时刻最佳预测值
        self.x = np.zeros((point_num, 4))
        # 定义每个关键点是否初始化
        self.first_frame_flag = np.ones(point_num)
        # 共预测的点数
        self.point_num = point_num

    """
    获取当前最佳预测值
    x_last、y_last：上一时刻最佳预测值
    x_observation、y_observation：这一时刻观测值
    i：当前预测的点
    """
    def predict(self, x_observation, y_observation, i):
        # 检测该关键点是否初始化
        if self.first_frame_flag[i]:
            self.x[id][0] = x_observation
            self.x[id][2] = y_observation
            self.first_frame_flag[i] = 0
        else:
            # 读取x_(k-1)
            self.x_mat[0, 0] = self.x[id][0]
            self.x_mat[2, 0] = self.x[id][2]

            x_predict = self.f_mat * self.x_mat  # x预估值
            p_predict = self.f_mat * self.p_mat * self.f_mat.T + self.q_mat  # x方差
            x_mat_before = self.x_mat  # 记录x_(k-1)

            kalman = p_predict * self.h_mat.T / (self.h_mat * p_predict * self.h_mat.T + self.r_mat)  # 计算卡尔曼权值
            # 通过观测值修正预测值
            x_mat = x_predict + np.multiply(kalman, (np.mat(
                [[x_observation, ], [x_observation - x_mat_before[0, 0], ], [y_observation, ],
                 [y_observation - x_mat_before[2, 0], ]]) - x_predict))
            # 将最佳预测写回x保存
            self.x[id][0] = x_mat[0, 0]
            self.x[id][2] = x_mat[2, 0]

            self.p_mat = (np.eye(x_mat.shape[0]) - kalman * self.h_mat) * p_predict
        # noise_x = x_observation + int(np.random.normal(0, 1) * 10)
        # noise_y = y_observation + int(np.random.normal(0, 1) * 10)

        # return self.x[id][0], self.x[id][2], noise_x, noise_y
        return self.x[id][0], self.x[id][2]


# 使用人体模型模型
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False,
                   model_complexity=1,
                   smooth_landmarks=True,
                   enable_segmentation=False,
                   smooth_segmentation=True,
                   min_detection_confidence=0.5,
                   min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
kalman = Kalman(33)

# 带有噪音观测点的样式
observation_landmarks_style  = mp_drawing.DrawingSpec(color=(255, 0, 0))
observation_connection_style = mp_drawing.DrawingSpec(color=(0, 0, 255))

# Kalman预测的样式
kalman_landmarks_style = mp_drawing.DrawingSpec(color=(0, 0, 255))
kalman_connection_style = mp_drawing.DrawingSpec(color=(0, 255, 0))


# 获取摄像头编号，笔记本摄像头默认为0
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    # opencv读出来的图片为gbr图片，使用cvtColor转变为rgb图片
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 结果侦测
    results = pose.process(img)

    h, w, c = img.shape

    # pose_landmarks返回人体姿态的33个点
    if results.pose_landmarks:

        # 打印带有噪音的观测值
        for id, lm in enumerate(results.pose_landmarks.landmark):
            # 计算2D关键点坐标
            cx, cy = int(lm.x * w), int(lm.y * h)
            cx += int(np.random.normal(0, 1) * 5)
            cy += int(np.random.normal(0, 1) * 5)


            lm.x = cx/w
            lm.y = cy/h
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS, observation_landmarks_style, observation_connection_style)

        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            # 获取卡尔曼滤波后的数值
            x_k, y_k = kalman.predict(cx, cy, id)
            # 写回landmark
            lm.x = x_k/w
            lm.y = y_k/h
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                  kalman_landmarks_style, kalman_connection_style)

    cv2.imshow('video', frame)

    # 执行退出
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
video.release()

```

![image-20220818152928175](.\mediapipe卡尔曼滤波算法的问题解答.assets\image-20220818152928175.png)

**蓝点红线**为带有噪音的**观测值**，**红点绿线**为通过卡尔曼滤波后的**最佳预测值**

## 6.尝试将针对3D点关键点检测平滑处理的卡尔曼滤波算法封装成函数接口

```python
class Kalman:
    # point_num:关键点的数量
    def __init__(self, point_num):
        # 定义预测变换矩阵
        self.x_mat = np.mat([[0, ], [0, ], [0, ], [0, ]])
        # 定义初始状态协方差矩阵
        self.p_mat = np.mat([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # 定义初始化控制矩阵
        self.b_mat = np.mat([[0.5, ], [1, ], [0.5, ], [1, ]])
        # 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
        self.f_mat = np.mat([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
        # 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
        self.q_mat = np.mat([[0.3, 0, 0, 0], [0, 0.3, 0, 0], [0, 0, 0.3, 0], [0, 0, 0, 0.3]])
        # 定义观测矩阵
        self.h_mat = np.mat([1, 0, 1, 0])
        # 定义观测噪声协方差
        self.r_mat = np.mat([1])
        # 定义存储上一个时刻最佳预测值
        self.x = np.zeros((point_num, 4))
        # 定义每个关键点是否初始化
        self.first_frame_flag = np.ones(point_num)
        # 共预测的点数
        self.point_num = point_num

    """
    获取当前最佳预测值
    x_last、y_last：上一时刻最佳预测值
    x_observation、y_observation：这一时刻观测值
    i：当前预测的点
    """
    def predict(self, x_observation, y_observation, i):
        # 检测该关键点是否初始化
        if self.first_frame_flag[i]:
            self.x[id][0] = x_observation
            self.x[id][2] = y_observation
            self.first_frame_flag[i] = 0
        else:
            # 读取x_(k-1)
            self.x_mat[0, 0] = self.x[id][0]
            self.x_mat[2, 0] = self.x[id][2]

            x_predict = self.f_mat * self.x_mat  # x预估值
            p_predict = self.f_mat * self.p_mat * self.f_mat.T + self.q_mat  # x方差
            x_mat_before = self.x_mat  # 记录x_(k-1)

            kalman = p_predict * self.h_mat.T / (self.h_mat * p_predict * self.h_mat.T + self.r_mat)  # 计算卡尔曼权值
            # 通过观测值修正预测值
            x_mat = x_predict + np.multiply(kalman, (np.mat(
                [[x_observation, ], [x_observation - x_mat_before[0, 0], ], [y_observation, ],
                 [y_observation - x_mat_before[2, 0], ]]) - x_predict))
            # 将最佳预测写回x保存
            self.x[id][0] = x_mat[0, 0]
            self.x[id][2] = x_mat[2, 0]

            self.p_mat = (np.eye(x_mat.shape[0]) - kalman * self.h_mat) * p_predict
        # noise_x = x_observation + int(np.random.normal(0, 1) * 10)
        # noise_y = y_observation + int(np.random.normal(0, 1) * 10)

        # return self.x[id][0], self.x[id][2], noise_x, noise_y
        return self.x[id][0], self.x[id][2]
```

## 7.了解卡尔曼滤波算法在计算机视觉领域中的其他应用

