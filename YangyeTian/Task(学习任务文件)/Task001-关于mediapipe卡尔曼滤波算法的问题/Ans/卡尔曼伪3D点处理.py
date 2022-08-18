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
