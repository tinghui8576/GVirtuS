# # Reference: https://medium.com/@Mert.A/how-to-use-yolov9-for-object-detection-93598ad88d7d
# import cv2
# from ultralytics import YOLO

# def predict(chosen_model, img, classes=[], conf=0.5):
#     if classes:
#         results = chosen_model.predict(img, classes=classes, conf=conf)
#     else:
#         results = chosen_model.predict(img, conf=conf)

#     return results

# def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
#     results = predict(chosen_model, img, classes, conf=conf)
#     for result in results:
#         for box in result.boxes:
#             cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
#                           (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
#             cv2.putText(img, f"{result.names[int(box.cls[0])]}",
#                         (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
#                         cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
#     return img, results

# model = YOLO("weights/yolov9e.pt")
# #.to('cuda')

# print("Model is on:", model.device)
# image = cv2.imread("images/FTP.6.62.6_Fruit-Yield-Back-Picture_1_2021-12-01-12-58-24.jpg")
# result_img, _ = predict_and_detect(model, image, classes=[], conf=0.17)
# cv2.imwrite("output.jpg", result_img)
# Reference: https://medium.com/@Mert.A/how-to-use-yolov9-for-object-detection-93598ad88d7d
import cv2
import time
from ultralytics import YOLO
device = 'cuda'
def predict(chosen_model, img, classes=[], conf=0.5):
    t0 = time.time()
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf, verbose=False, device = device)
    else:
        results = chosen_model.predict(img, conf=conf, verbose=False, device = device)
    t1 = time.time()
    with open("inference_times.txt", "a") as f:
        f.write(f"{(t1 - t0) * 1000:.2f}\n")
    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

model = YOLO("weights/yolov9e.pt")
model.to(device)
with open("inference_times.txt", "w") as f:
    f.write(f"Inference time\n")
for i in range(100):
    image = cv2.imread("images/FTP.6.62.6_Fruit-Yield-Back-Picture_1_2021-12-01-12-58-24.jpg")
    result_img, _ = predict_and_detect(model, image, classes=[], conf=0.17)
cv2.imwrite("output.jpg", result_img)