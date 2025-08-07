import requests
import time
from multiprocessing import Pool

def send_request(i):
    prompt = f"Request {i}: Give me a short intro to large language models."
    send_time = time.time()

    response = requests.post("http://localhost:8000/generate", json={"prompt": prompt})
    resp_json = response.json()

    receive_time = time.time()

    # Extract server-side timing if available
    timing = resp_json.get("timing", {})
    server_inference_time = timing.get("inference_duration", None)
    total_latency = timing.get("total_latency", None)

    # Print everything nicely
    result = (
        f"[Request {i}]\n"
        f"Client send time:     {send_time:.6f}\n"
        f"Client receive time:  {receive_time:.6f}\n"
        f"Round-trip time:      {receive_time - send_time:.2f} s\n"
    )

    if timing:
        result += (
            f"Server arrival time:  {timing['arrival']:.6f}\n"
            f"Server serve start:   {timing['serving_start']:.6f}\n"
            f"Server inference:     {server_inference_time:.2f} s\n"
            f"Server total latency: {total_latency:.2f} s\n"
        )

    result += f"Response: {resp_json['response']}\n{'-'*60}\n"
    return result

if __name__ == "__main__":
    with Pool(processes=10) as pool:
        results = pool.map(send_request, range(10))

    for res in results:
        print(res)
