import os
import time
import torch
from torchvision import models
from PIL import Image

model_selectors = [
    "squeezenet1_1",
#    "mobilenet_v2",
#    "resnet18",
#    "resnet50",
#    "resnet101",
#    "resnet152",
#    "vgg16",
#    "vgg19",
]

model_registry = {
    "squeezenet1_1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.DEFAULT),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
    "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
    "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
    "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT),
}

device = "cuda" 
#device = "cpu"

img_dir = "imagenet_test_1000"
batch_sizes = [1, 10, 100, 200]
#batch_sizes = [1]

for model_selector in model_selectors:
    if model_selector not in model_registry:
        print(f"Model '{model_selector}' is not supported, skipping.")
        continue

    model_fn, weights = model_registry[model_selector]

    if model_selector == "inception_v3":
        model = model_fn(weights=weights, aux_logits=False)
    else:
        model = model_fn(weights=weights)

    model = model.to(device)
    model.eval()

    preprocess = weights.transforms()

    for batch_size in batch_sizes:
        time_log_file = f"time_{model_selector}_bs{batch_size}_{device}.txt"
        correct = 0
        total = 0

        images = []
        labels = []
        filenames = []

        with open(time_log_file, "w") as f:
            file_list = sorted(os.listdir(img_dir))
            for idx, filename in enumerate(file_list):
                if not (filename.endswith(".JPEG") or filename.endswith(".jpg") or filename.endswith(".png")):
                    continue

                true_label = int(filename.split("_")[0])

                img_path = os.path.join(img_dir, filename)
                img = Image.open(img_path).convert("RGB")
                input_tensor = preprocess(img)
                images.append(input_tensor)
                labels.append(true_label)
                filenames.append(filename)

                # batch 满了或者最后一张
                if len(images) == batch_size or idx == len(file_list) - 1:
                    input_batch = torch.stack(images).to(device)

                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.time()
                    with torch.no_grad():
                        outputs = model(input_batch)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    end = time.time()
                    elapsed = (end - start) * 1000  # ms per batch
                    elapsed_per_img = elapsed / len(images)

                    # 获取预测结果
                    _, predicted_idxs = torch.max(outputs, 1)

                    for j, predicted_label in enumerate(predicted_idxs):
                        if predicted_label.item() == labels[j]:
                            correct += 1
                        total += 1
                        f.write(f"{filenames[j]}\t{elapsed_per_img:.4f} ms\n")

                    # 清空 batch
                    images = []
                    labels = []
                    filenames = []

        # Accuracy
        accuracy = 100.0 * correct / total
        print(f"[{model_selector} | batch={batch_size}] Accuracy on {total} images: {accuracy:.2f}%")
        print(f"[{model_selector} | batch={batch_size}] Per-image inference times saved to {time_log_file}")

