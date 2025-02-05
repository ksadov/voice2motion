import os
import requests

CHECKPOINT_URL = "https://huggingface.co/cherrvak/fp16_0.2_constant/resolve/main/epoch_24_step_80000.pt"
MEDIAPIPE_FILE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


def download_from_url(url, filename):
    os.makedirs("assets", exist_ok=True)
    if not os.path.exists(f"assets/{filename}"):
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url)
        with open(f"assets/{filename}", "wb") as f:
            f.write(response.content)


def download_files():
    download_from_url(MEDIAPIPE_FILE_URL, "face_landmarker.task")
    download_from_url(CHECKPOINT_URL, "demo_checkpoint.pt")


if __name__ == "__main__":
    download_files()
