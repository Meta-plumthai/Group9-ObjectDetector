import cv2
import time
import os
from src.detector import ObjectDetectorModel

model_path = "models/efficientdet_lite0.tflite"
if not os.path.exists(model_path):
    print(f"[Error] ไม่พบไฟล์โมเดลที่: {os.path.abspath(model_path)}")
    exit()

print("[1/3] กำลังโหลดโมเดล...")
detector = ObjectDetectorModel(
    model_path=model_path,
    score_threshold=0.3,
    max_results=5
)
print("[2/3] โหลดโมเดลสำเร็จ!")

# เปิดกล้องโดยไม่บังคับ CAP_DSHOW (ใช้ Index 0)
print("[3/3] กำลังเปิดกล้อง...")
cap = cv2.VideoCapture(0)

# ถ้ากล้องยังเปิดไม่ได้ ให้ลอง Index 1
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

# ปรับความละเอียดกล้องมาตรฐาน
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# รอให้กล้อง Warm up 1 วินาที
time.sleep(1)

print("-> เปิดกล้องสำเร็จ! กดปุ่ม 'q' ที่หน้าต่างภาพเพื่อปิดโปรแกรม")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # คำนวณ Inference
    start_time = time.time()
    results = detector.detect(frame)
    inference_time = (time.time() - start_time) * 1000

    # ตีกรอบผลลัพธ์
    if results.detections:
        for detection in results.detections:
            category = detection.categories[0]
            bbox = detection.bounding_box

            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            w = int(bbox.width)
            h = int(bbox.height)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{category.category_name} ({category.score * 100:.1f}%)"
            cv2.putText(
                frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # แสดง Inference Time
    cv2.putText(
        frame,
        f"Inference: {inference_time:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("MediaPipe Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


if __name__ == "__main__":
    import numpy as np
    
    print("กำลังทดสอบสร้าง ObjectDetectorModel...")
    detector = ObjectDetectorModel()
    print("โหลด Model สำเร็จ!")
    
    # จำลองรูปภาพสีดำขนาด 480x640 เพื่อทดสอบระบบประมวลผล
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    raw_res = detector.detect(dummy_frame)
    parsed_res = detector.to_dict_list(raw_res)
    print("ทดสอบรันสำเร็จ ผลลัพธ์ที่ได้:", parsed_res)