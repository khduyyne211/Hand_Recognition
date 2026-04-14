import os
import cv2
import csv
import mediapipe as mp
import kagglehub

# Tải dataset
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
dataset_path = os.path.join(path, "asl_alphabet_train", "asl_alphabet_train")

# Khởi tạo MediaPipe chế độ ảnh tĩnh
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Chữ cái cần xử lý (bỏ J và Z)
labels = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]

# Số mẫu tối đa mỗi chữ
max_per_label = 200

print("Bắt đầu trích xuất landmark...")

with open("data/data.csv", "a", newline="") as f:
    writer = csv.writer(f)

    for label in labels:
        folder = os.path.join(dataset_path, label)

        if not os.path.exists(folder):
            print(f"Không tìm thấy thư mục: {label}")
            continue

        images = os.listdir(folder)[:max_per_label]
        count = 0

        for img_file in images:
            img_path = os.path.join(folder, img_file)
            image = cv2.imread(img_path)

            if image is None:
                continue

            # Chuyển BGR → RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    writer.writerow(landmarks + [label])
                    count += 1
                    break

        print(f"✅ {label}: {count} mẫu")

hands.close()
print("Hoàn thành! Kiểm tra file data/data.csv")