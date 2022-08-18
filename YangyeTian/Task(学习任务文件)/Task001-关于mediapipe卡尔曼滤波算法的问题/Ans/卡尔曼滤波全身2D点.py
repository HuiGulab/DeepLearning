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
