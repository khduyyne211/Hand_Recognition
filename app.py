import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from collections import deque

# =============================================
# 1. LOAD MODEL
# =============================================
print("Đang tải model...")
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
print("Tải model thành công!")

# =============================================
# 2. KHỞI TẠO MEDIAPIPE & CAMERA
# =============================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
cap = cv2.VideoCapture(0)

# =============================================
# 3. NHẬN DẠNG J VÀ Z BẰNG QUỸ ĐẠO
# =============================================
wrist_history = deque(maxlen=15)

def detect_J(history):
    if len(history) < 10:
        return False
    points = list(history)
    dx = points[-1][0] - points[0][0]
    dy = points[-1][1] - points[0][1]
    return dy > 0.04 and dx < -0.03

def detect_Z(history):
    if len(history) < 12:
        return False
    points = list(history)
    direction_changes = 0
    prev_dir = None
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        if abs(dx) < 0.005:
            continue
        curr_dir = 1 if dx > 0 else -1
        if prev_dir is not None and curr_dir != prev_dir:
            direction_changes += 1
        prev_dir = curr_dir
    return direction_changes >= 2

# =============================================
# 4. CÁC BIẾN THEO DÕI
# =============================================
sentence      = ""
current_letter = ""
hold_start_time    = None
hand_absent_time   = None
space_added        = False
last_confirmed_time = 0

HOLD_TIME            = 1.0   # Giữ tư thế 1 giây
ABSENT_TIME          = 0.8   # Rút tay 0.8 giây → thêm space
COOLDOWN_TIME        = 0.5   # Chờ giữa 2 chữ
CONFIDENCE_THRESHOLD = 0.85  # Độ tin cậy tối thiểu

print("\nỨng dụng ASL đã khởi động!")
print("Hỗ trợ 26 chữ cái A-Z (bao gồm J và Z)")
print("Phím tắt: BACKSPACE=xóa chữ | C=xóa câu | Q=thoát\n")

# =============================================
# 5. VÒNG LẶP CHÍNH
# =============================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    now    = time.time()
    h, w   = frame.shape[:2]

    if result.multi_hand_landmarks:
        hand_absent_time = None
        space_added      = False

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lưu vị trí cổ tay để nhận J/Z
            wrist = hand_landmarks.landmark[0]
            wrist_history.append((wrist.x, wrist.y))

            # Trích xuất 63 landmark
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Dự đoán chữ
            prediction = model.predict([landmarks])[0]
            confidence = max(model.predict_proba([landmarks])[0])
            current_letter = prediction

            # Tính thời gian giữ tư thế
            if hold_start_time is None:
                hold_start_time = now
            hold_duration = now - hold_start_time
            cooldown_ok   = (now - last_confirmed_time) > COOLDOWN_TIME

            # --- Kiểm tra J và Z ---
            is_J = detect_J(wrist_history)
            is_Z = detect_Z(wrist_history)

            if is_J and cooldown_ok:
                sentence += "J"
                last_confirmed_time = now
                wrist_history.clear()
                current_letter = "J"

            elif is_Z and cooldown_ok:
                sentence += "Z"
                last_confirmed_time = now
                wrist_history.clear()
                current_letter = "Z"

            # --- Xác nhận chữ thường (giữ 1 giây) ---
            elif hold_duration >= HOLD_TIME and cooldown_ok:
                if confidence >= CONFIDENCE_THRESHOLD:
                    sentence += prediction
                    last_confirmed_time = now
                    hold_start_time     = now

            # --- Vẽ thanh tiến trình ---
            progress  = min(hold_duration / HOLD_TIME, 1.0)
            bar_width = int(300 * progress)

            if confidence >= CONFIDENCE_THRESHOLD:
                bar_color = (0, 255, 0)    # Xanh lá
            elif confidence >= 0.7:
                bar_color = (0, 165, 255)  # Cam
            else:
                bar_color = (0, 0, 255)    # Đỏ

            cv2.rectangle(frame, (10, 80), (310, 100), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 80), (10 + bar_width, 100), bar_color, -1)

            # --- Hiện chữ + confidence ---
            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD \
                    else (0, 165, 255)
            cv2.putText(frame,
                        f"Nhan dang: {current_letter} ({confidence*100:.0f}%)",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # --- Thông báo J/Z ---
            if is_J:
                cv2.putText(frame, ">> Phat hien chu J! <<",
                            (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif is_Z:
                cv2.putText(frame, ">> Phat hien chu Z! <<",
                            (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- Vẽ quỹ đạo cổ tay (màu tím) ---
            pts = list(wrist_history)
            for i in range(1, len(pts)):
                x1, y1 = int(pts[i-1][0] * w), int(pts[i-1][1] * h)
                x2, y2 = int(pts[i][0]   * w), int(pts[i][1]   * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

    else:
        # Không thấy tay
        current_letter  = ""
        hold_start_time = None
        wrist_history.clear()

        cv2.putText(frame, "Dang cho ban tay...",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Tự động thêm SPACE khi rút tay
        if not space_added:
            if hand_absent_time is None:
                hand_absent_time = now
            elif now - hand_absent_time >= ABSENT_TIME:
                if len(sentence) > 0 and sentence[-1] != " ":
                    sentence += " "
                    space_added = True

    # =============================================
    # 6. HIỂN THỊ GIAO DIỆN
    # =============================================

    # Header đen phía trên hiện câu
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)

    # Hiện câu (nếu dài thì cắt bớt từ đầu)
    display_sentence = sentence[-35:] if len(sentence) > 35 else sentence
    cv2.putText(frame, f"Cau: {display_sentence}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Hướng dẫn phía dưới
    cv2.putText(frame, "BS=xoa chu | C=xoa cau | Q=thoat",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("ASL Hand Sign Recognition", frame)

    # =============================================
    # 7. XỬ LÝ PHÍM BẤM
    # =============================================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 8:        # BACKSPACE
        sentence = sentence[:-1]
    elif key == ord('c'): # Xóa câu
        sentence = ""

# =============================================
# 8. GIẢI PHÓNG TÀI NGUYÊN
# =============================================
cap.release()
cv2.destroyAllWindows()
print("Đã thoát ứng dụng!")