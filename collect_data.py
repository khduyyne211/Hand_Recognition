import cv2
import mediapipe as mp
import csv
import os

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Khởi tạo Camera
cap = cv2.VideoCapture(0)

# Hỏi người dùng muốn thu chữ gì
label = input("Nhập chữ cái muốn thu thập (A-Z): ").upper()
count = 0
max_samples = 150

print(f"Chuẩn bị thu thập {max_samples} mẫu cho chữ '{label}'...")
print("Nhấn Q để dừng sớm")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Lật camera để tự nhiên hơn
    frame = cv2.flip(frame, 1)

    # Chuyển sang RGB vì MediaPipe cần RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe xử lý frame
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Vẽ landmark lên màn hình
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            # Làm phẳng 21 điểm → 63 số
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Lưu vào CSV
            file_exists = os.path.exists("data/data.csv")
            with open("data/data.csv", "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = [f"{axis}{i}"
                             for i in range(21)
                             for axis in ["x", "y", "z"]]
                    header.append("label")
                    writer.writerow(header)
                writer.writerow(landmarks + [label])

            count += 1

    # Hiển thị thông tin lên màn hình
    cv2.putText(frame, f"Thu: {label} | Mau: {count}/{max_samples}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("Thu thap du lieu", frame)

    # Đủ {max_samples} mẫu → dừng tự động
    if count >= max_samples:
        print(f"✅ Đã thu đủ {max_samples} mẫu cho chữ '{label}'!")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()