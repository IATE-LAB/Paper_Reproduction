import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# 模型路径
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# 情绪类别
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# 加载模型
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# 视频文件路径
video_path = '1.mp4'  # 替换为你的视频文件路径
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 缩放比例
scale = 0.6  # 比例可以根据需求调整
new_width = int(frame_width * scale)
new_height = int(frame_height * scale)

# 逐帧处理视频
while True:
    ret, frame = cap.read()
    if not ret:
        break  # 视频读取结束

    # 缩放视频帧
    frame = cv2.resize(frame, (new_width, new_height))

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # 创建空白画布显示概率
    canvas = np.zeros((200, 300, 3), dtype="uint8")  # 缩小画布尺寸
    frameClone = frame.copy()

    if len(faces) > 0:
        # 按面积排序，选择最大的脸部区域
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        # 处理脸部区域
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # 预测情绪
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        # 绘制结果
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 25) + 5), (w, (i * 25) + 25), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 25) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                      (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Emotion Recognition', frameClone)
    cv2.imshow("Probabilities", canvas)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
