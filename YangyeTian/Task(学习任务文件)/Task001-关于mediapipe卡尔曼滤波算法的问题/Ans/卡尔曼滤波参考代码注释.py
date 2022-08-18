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
        print(results.multi_hand_landmarks)
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
