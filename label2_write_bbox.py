import os
import cv2

# =========================
# CẤU HÌNH
# =========================
VIDEO_PATH = r"E:\A_HK8\Video_New_123\test3\img\3.mp4"
TRACKER_TXT_DIR = r"E:\A_HK8\Video_New_123\test3\video_track\jetson_nano_pure_predecoded_dualacc_txt"
YOLO_TXT_DIR = r"E:\A_HK8\Video_New_123\new3\obj_train_data"
OUTPUT_PATH = r"E:\A_HK8\Video_New_123\3_output.mp4"

TRACKER_NAME = "tracker"
YOLO_NAME = "object"

TRACKER_COLOR = (0, 0, 255)   # đỏ
YOLO_COLOR = (0, 255, 0)      # xanh lá

BOX_THICKNESS = 2
FONT_SCALE = 0.8
FONT_THICKNESS = 2

# Ảnh hiện tại đang bị nằm ngang.
# Để đưa người về tư thế đứng bình thường, xoay 90 độ sang phải.
ROTATE_FLAG = cv2.ROTATE_90_CLOCKWISE


# =========================
# HÀM TẠO ĐƯỜNG DẪN FILE TXT
# =========================
def make_txt_path(folder, frame_idx):
    return os.path.join(folder, f"frame_{frame_idx:06d}.txt")


# =========================
# XOAY FRAME
# =========================
def rotate_frame(frame):
    return cv2.rotate(frame, ROTATE_FLAG)


# =========================
# ĐỌC FILE TRACKER TXT
# Format:
# frame_index=0
# bbox_xywh=212.184288,257.928192,144.936000,457.832448
# time_s=41.476887
# fps=0.024110
# =========================
def read_tracker_txt(txt_path):
    if not os.path.exists(txt_path):
        return None

    data = {
        "frame_index": None,
        "bbox_xywh": None,
        "time_s": None,
        "fps": None,
    }

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "frame_index":
                data["frame_index"] = int(value)

            elif key == "bbox_xywh":
                parts = value.split(",")
                if len(parts) == 4:
                    x = float(parts[0])
                    y = float(parts[1])
                    w = float(parts[2])
                    h = float(parts[3])
                    data["bbox_xywh"] = (x, y, w, h)

            elif key == "time_s":
                data["time_s"] = float(value)

            elif key == "fps":
                data["fps"] = float(value)

    return data


# =========================
# ĐỌC FILE YOLO TXT
# Format:
# 0 0.500564 0.475728 0.376302 0.443467
# class xc yc w h
# =========================
def read_yolo_txt(txt_path):
    if not os.path.exists(txt_path):
        return []

    objects = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                objects.append((class_id, xc, yc, w, h))
            except ValueError:
                continue

    return objects


# =========================
# TRACKER bbox_xywh -> xyxy
# x, y là góc trái trên, w h là kích thước pixel
# =========================
def tracker_xywh_to_xyxy(x, y, w, h, img_w, img_h):
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2


# =========================
# YOLO normalized -> xyxy
# =========================
def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    box_w = w * img_w
    box_h = h * img_h
    center_x = xc * img_w
    center_y = yc * img_h

    x1 = int(round(center_x - box_w / 2.0))
    y1 = int(round(center_y - box_h / 2.0))
    x2 = int(round(center_x + box_w / 2.0))
    y2 = int(round(center_y + box_h / 2.0))

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2


# =========================
# VẼ 1 BBOX + NHÃN
# =========================
def draw_box_with_label(frame, x1, y1, x2, y2, color, label_text):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    (tw, th), baseline = cv2.getTextSize(
        label_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_THICKNESS
    )

    text_x1 = x1
    text_y1 = max(0, y1 - th - baseline - 4)
    text_x2 = x1 + tw + 8
    text_y2 = y1

    cv2.rectangle(frame, (text_x1, text_y1), (text_x2, text_y2), color, -1)

    cv2.putText(
        frame,
        label_text,
        (x1 + 4, max(th + 2, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


# =========================
# VẼ THÔNG TIN FRAME/FPS
# dựa trên tracker txt
# =========================
def draw_info(frame, frame_idx, fps_value):
    line1 = f"Frame: {frame_idx}"
    line2 = f"FPS: {fps_value:.3f}" if fps_value is not None else "FPS: N/A"

    cv2.putText(
        frame, line1, (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 255, 255), 2, cv2.LINE_AA
    )

    cv2.putText(
        frame, line2, (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (0, 255, 255), 2, cv2.LINE_AA
    )


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Không tìm thấy video: {VIDEO_PATH}")

    if not os.path.isdir(TRACKER_TXT_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục tracker txt: {TRACKER_TXT_DIR}")

    if not os.path.isdir(YOLO_TXT_DIR):
        raise FileNotFoundError(f"Không tìm thấy thư mục yolo txt: {YOLO_TXT_DIR}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {VIDEO_PATH}")

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Đọc thử frame đầu để biết kích thước sau khi xoay
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Không đọc được frame đầu tiên của video.")

    first_frame = rotate_frame(first_frame)
    out_h, out_w = first_frame.shape[:2]

    print(f"[INFO] Video FPS: {fps_video}")
    print(f"[INFO] Total frames: {total_frames}")
    print(f"[INFO] Output size after rotation: {out_w}x{out_h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (out_w, out_h))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không tạo được output video: {OUTPUT_PATH}")

    # Quay lại đầu video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    missing_tracker = 0
    missing_yolo = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) Xoay frame trước để về đúng chiều đứng
        frame = rotate_frame(frame)
        frame_h, frame_w = frame.shape[:2]

        # 2) Đọc tracker txt
        tracker_txt_path = make_txt_path(TRACKER_TXT_DIR, frame_idx)
        tracker_data = read_tracker_txt(tracker_txt_path)

        tracker_fps = None
        if tracker_data is None:
            missing_tracker += 1
        else:
            tracker_fps = tracker_data.get("fps", None)
            bbox_xywh = tracker_data.get("bbox_xywh", None)

            if bbox_xywh is not None:
                tx, ty, tw, th = bbox_xywh
                x1, y1, x2, y2 = tracker_xywh_to_xyxy(tx, ty, tw, th, frame_w, frame_h)
                draw_box_with_label(frame, x1, y1, x2, y2, TRACKER_COLOR, TRACKER_NAME)

        # 3) Đọc yolo txt
        yolo_txt_path = make_txt_path(YOLO_TXT_DIR, frame_idx)
        yolo_objects = read_yolo_txt(yolo_txt_path)

        if len(yolo_objects) == 0:
            missing_yolo += 1
        else:
            for class_id, xc, yc, w, h in yolo_objects:
                x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, frame_w, frame_h)
                draw_box_with_label(frame, x1, y1, x2, y2, YOLO_COLOR, YOLO_NAME)

        # 4) Vẽ frame + fps
        display_frame_idx = frame_idx
        if tracker_data is not None and tracker_data.get("frame_index") is not None:
            display_frame_idx = tracker_data["frame_index"]

        draw_info(frame, display_frame_idx, tracker_fps)

        # 5) Ghi video
        writer.write(frame)

        if frame_idx % 100 == 0:
            print(
                f"[INFO] Frame {frame_idx}/{total_frames} | "
                f"tracker_txt={'OK' if tracker_data is not None else 'MISS'} | "
                f"yolo_txt={'OK' if len(yolo_objects) > 0 else 'MISS'}"
            )

            if frame_idx == 0:
                print(f"[DEBUG] Rotated frame size = {frame_w}x{frame_h}")
                if tracker_data is not None:
                    print(f"[DEBUG] Tracker frame 0: {tracker_data}")
                if len(yolo_objects) > 0:
                    print(f"[DEBUG] YOLO frame 0: {yolo_objects}")

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"[DONE] Xuất xong video: {OUTPUT_PATH}")
    print(f"[DONE] Số frame thiếu tracker txt: {missing_tracker}")
    print(f"[DONE] Số frame thiếu yolo txt: {missing_yolo}")


if __name__ == "__main__":
    main()