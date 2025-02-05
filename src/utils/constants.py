import os


def exact_div(x, y):
    assert x % y == 0
    return x // y


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
# 3000 frames in a mel spectrogram input
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token
TIMESTEP_S = 30 / 1500

VIDEO_FPS = 30
N_AUDIO_SAMPLES_PER_VIDEO_FRAME = SAMPLE_RATE // VIDEO_FPS
N_VIDEO_FRAMES = CHUNK_LENGTH * VIDEO_FPS  # 900 frames in a 30-second video chunk


def mel_frames_from_video_frames(n_video_frames):
    return int(n_video_frames * N_SAMPLES_PER_TOKEN / VIDEO_FPS)


MEL_FILTER_PATH = os.path.join(
    os.path.dirname(__file__), "../../assets", "mel_filters.npz"
)
LANDMARKER_PATH = "assets/face_landmarker.task"

BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]

HEAD_ANGLE_NAMES = ["pitch", "yaw", "roll"]

HEAD_LANDMARK_DIM = len(BLENDSHAPE_NAMES) + len(HEAD_ANGLE_NAMES)


def get_n_mels(whisper_model_name: str):
    if "v3" in whisper_model_name:
        return 128
    return 80
