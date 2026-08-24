import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ObjectDetectorModel:
    def __init__(self, model_path: str = "models/efficientdet_lite0.tflite", score_threshold: float = 0.5, max_results: int = 3):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=score_threshold,
            max_results=max_results,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect(self, image_np):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        return self.detector.detect(mp_image)

    def to_dict_list(self, detection_result):
        results = []
        if not detection_result or not detection_result.detections:
            return results

        for detection in detection_result.detections:
            category = detection.categories[0]
            bbox = detection.bounding_box
            results.append({
                "label": category.category_name,
                "score": float(category.score),
                "box": {
                    "x": int(bbox.origin_x),
                    "y": int(bbox.origin_y),
                    "width": int(bbox.width),
                    "height": int(bbox.height)
                }
            })
        return results

# เอาบล็อกทดสอบมาไว้ไฟล์นี้
if __name__ == "__main__":
    import numpy as np
    
    print("กำลังทดสอบสร้าง ObjectDetectorModel...")
    # ชี้ path ให้ถูกต้องเวลาทดสอบจากในโฟลเดอร์ src
    detector = ObjectDetectorModel(model_path="../models/efficientdet_lite0.tflite") 
    print("โหลด Model สำเร็จ!")
    
    # จำลองรูปภาพสีดำขนาด 480x640 เพื่อทดสอบ
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    raw_res = detector.detect(dummy_frame)
    parsed_res = detector.to_dict_list(raw_res)
    print("ทดสอบรันสำเร็จ ผลลัพธ์ที่ได้:", parsed_res)