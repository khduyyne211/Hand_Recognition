import cv2
import mediapipe as mp
import csv
import os
import time
import utils

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

# Cấu hình các giai đoạn
stages = [
    {"name": "THANG",         "instruction": "Giu tay THANG truoc camera",    "samples": 30},
    {"name": "NGHIENG TRAI",  "instruction": "Nghieng tay sang TRAI",         "samples": 30},
    {"name": "NGHIENG PHAI",  "instruction": "Nghieng tay sang PHAI",         "samples": 30},
    {"name": "NGHIENG LEN",   "instruction": "Nghieng tay LEN TREN",          "samples": 30},
    {"name": "NGHIENG XUONG", "instruction": "Nghieng tay XUONG DUOI",        "samples": 30},
]

total_samples = sum(s["samples"] for s in stages)
current_stage = 0
stage_count = 0
total_count = 0
COUNTDOWN = 3  

print(f"\nChuẩn bị thu thập {total_samples} mẫu cho chữ '{label}'")
print(f"Gồm {len(stages)} giai đoạn, mỗi giai đoạn {stages[0]['samples']} mẫu")
print("Nhấn Q để dừng sớm\n")

# Bắt đầu đếm ngược lần đầu tiên ngay trước khi vào vòng lặp
countdown_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    now = time.time()
    stage = stages[current_stage]

    # === XỬ LÝ ĐẾM NGƯỢC ===
    if countdown_start is not None:
        elapsed = now - countdown_start
        remaining = int(COUNTDOWN - elapsed) + 1

        if elapsed < COUNTDOWN:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, f"Giai doan {current_stage + 1}/{len(stages)}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.putText(frame, stage["instruction"], (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Chuan bi: {remaining}", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

            cv2.imshow("Thu thap du lieu", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        else:
            # Đếm ngược xong -> Reset biến để đi thẳng vào thu thập
            countdown_start = None

    # === THU THẬP DỮ LIỆU ===
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = utils.process_landmarks(hand_landmarks)

            file_exists = os.path.exists("data/data_custom.csv")
            with open("data/data_custom.csv", "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    header = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
                    header.append("label")
                    writer.writerow(header)
                writer.writerow(landmarks + [label])

            stage_count += 1
            total_count += 1

    # === HIỂN THỊ THÔNG TIN ===
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
    cv2.putText(frame, f"Chu: {label} | Tong: {total_count}/{total_samples}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Giai doan {current_stage + 1}/{len(stages)}: {stage['name']}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"  {stage['instruction']} ({stage_count}/{stage['samples']})", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    bar_total = int((total_count / total_samples) * (frame.shape[1] - 20))
    cv2.rectangle(frame, (10, 115), (frame.shape[1] - 10, 130), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 115), (10 + bar_total, 130), (0, 255, 0), -1)

    # === KIỂM TRA HOÀN THÀNH GIAI ĐOẠN ===
    if stage_count >= stage["samples"]:
        current_stage += 1
        stage_count = 0

        if current_stage >= len(stages):
            print(f"\n✅ Đã thu đủ {total_samples} mẫu cho chữ '{label}'!")
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, "HOAN THANH!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            cv2.imshow("Thu thap du lieu", frame)
            cv2.waitKey(2000)
            break
        else:
            print(f"✅ Giai đoạn {current_stage} xong! Chuẩn bị: {stages[current_stage]['name']}")
            # Khởi động lại đếm ngược cho giai đoạn tiếp theo
            countdown_start = time.time() 

    cv2.imshow("Thu thap du lieu", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()