关于mediapipe卡尔曼滤波算法的问题

1.请描述卡尔曼滤波算法的基本原理

2.附件代码已实现卡尔曼滤波算法对单个2D关键点的实时平滑处理

3.请将卡尔曼滤波算法扩展到mediapipe全身2D关键点并总结实验效果

4.请验证mediapipe自带的伪3D关键点检测与2D关键点检测的x，y坐标是否一致？

5.尝试将卡尔曼滤波算法扩展到mediapipe伪3D关键点的平滑处理中，并总结实验效果

6.尝试将针对3D点关键点检测平滑处理的卡尔曼滤波算法封装成函数接口

7.了解卡尔曼滤波算法在计算机视觉领域中的其他应用



附件：

https://blog.csdn.net/qq_42500340/article/details/124476348

参考代码如下：

```python
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

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

video = cv2.VideoCapture(0)
first_frame_flag = True
while True:
    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img)
    h, w, c = img.shape
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 9:
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
                        cv2.circle(frame, (noise_x, noise_y), 6, (0, 0, 255), cv2.FILLED)
                        cv2.circle(frame, (int(x_mat[0, 0]), int(x_mat[2, 0])), 3, (0, 255, 0), cv2.FILLED)
                    break
    cv2.imshow('video', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
video.release()
```

