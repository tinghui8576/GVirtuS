import os
import time
import torch
from torchvision import models
from PIL import Image

model_selectors = [
    "squeezenet1_1",
    # "mobilenet_v2",
    # "resnet18",
    # "vgg16",
]

model_registry = {
    "squeezenet1_1": (models.squeezenet1_1, models.SqueezeNet1_1_Weights.DEFAULT),
    # "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
    # "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    # "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT),
}

device = "cuda" 
# device="cpu"

img_dir = "imagenet_test_1000"
input_shape = (1, 3, 224, 224)   
dtype = torch.float32 

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

    time_log_file = f"time_{model_selector}_{device}_graph.txt"

    correct = 0
    total = 0

    Warmup = 3
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    static_input = torch.empty(input_shape, device=device, dtype=dtype)
    static_output = None
    with open(time_log_file, "w") as f:
        for filename in sorted(os.listdir(img_dir)):
            if not (filename.endswith(".JPEG") or filename.endswith(".jpg") or filename.endswith(".png")):
                continue

            true_label = int(filename.split("_")[0])

            img_path = os.path.join(img_dir, filename)
            img = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img)
            input_batch = input_tensor.unsqueeze(0).to(device, non_blocking=True)

            static_input.copy_(input_batch)
            # Inference timing
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            if Warmup > 0:
                with torch.cuda.stream(s):
                    with torch.no_grad():
                        static_output = model(static_input)
                torch.cuda.current_stream().wait_stream(s)
            elif Warmup == 0:
                g = torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with torch.cuda.stream(s):
                    g.capture_begin()
                    static_output = model(static_input)
                    g.capture_end()
                torch.cuda.synchronize()
            else:
                g.replay()
            # static_output = model(static_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            elapsed = (end - start) * 1000  # ms
            
            # Prediction
            _, predicted_idx = torch.max(static_output, 1)
            predicted_label = predicted_idx.item()

            if predicted_label == true_label:
                correct += 1
            total += 1
            Warmup -= 1

            # Save time log
            f.write(f"{filename}\t{elapsed:.4f} ms\n")

    # Accuracy
    accuracy = 100.0 * correct / total
    print(f"[{model_selector}] Accuracy on {total} images: {accuracy:.2f}%")
    print(f"[{model_selector}] Per-image inference times saved to {time_log_file}")
