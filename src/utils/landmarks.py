from re import T
from weakref import ref
import cv2
import math
import torch
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Tuple, List, Union
from scipy.spatial.transform import Rotation

from src.utils.constants import LANDMARKER_PATH, BLENDSHAPE_NAMES


def scale_and_center_head_angles(
    head_angles: NDArray[np.float32],
    bad_frames: List[int] = [],
    just_center: bool = False,
) -> NDArray[np.float32]:
    """
    Head angles are between -180 and 180. Center and scale them to be in range [-1, 1].

    Args:
        head_angles: Sequence of pitch, yaw, roll values of shape (temporal_dim, 3)
        bad_frames: List of indices of frames where face detection failed

    Returns:
        Array of scaled and centered head angles of shape (temporal_dim, 3)
    """
    good_frames = [i for i in range(head_angles.shape[0]) if i not in bad_frames]
    head_angles_mean = head_angles[good_frames].mean(axis=0)
    head_angles[good_frames] = head_angles[good_frames] - head_angles_mean
    if not just_center:
        head_angles[good_frames] = head_angles[good_frames] / 180.0
    return head_angles


def unscale_and_uncenter_head_angles(
    head_angles: NDArray[np.float32],
    mean_pos: Optional[NDArray[np.float32]] = None,
    bad_frames: List[int] = [],
) -> NDArray[np.float32]:
    """
    Rescale head angles in range [-1, 1] to [-180, 180] and offset by mean position.

    Args:
        head_angles: Sequence of pitch, yaw, roll values of shape (temporal_dim, 3)
        mean_pos: Mean position to offset the head angles of shape (3,)
        bad_frames: List of indices of frames where face detection failed

    Returns:
        Array of unscaled and uncentered head angles of shape (temporal_dim, 3)
    """
    if mean_pos is None:
        mean_pos = np.zeros(3).astype(np.float32)
    good_frames = [i for i in range(head_angles.shape[0]) if i not in bad_frames]
    head_angles[good_frames] = head_angles[good_frames] + mean_pos
    head_angles[good_frames] = head_angles[good_frames] * 180.0
    return head_angles


def init_face_landmarker(delegate: str) -> FaceLandmarker:
    """
    Initialize the face landmarker model.

    Args:
        delegate: Device to run the model on (CPU or GPU)

    Returns:
        FaceLandmarker model
    """
    base_options = BaseOptions(model_asset_path=LANDMARKER_PATH, delegate=delegate)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def get_frames_from_video(video_path: Path) -> NDArray[np.uint8]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Array of frames of shape (temporal_dim, height, width, channels)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


def info_for_frame(
    frames: NDArray[np.uint8], frame_idx: int, detector: FaceLandmarker
) -> Tuple[
    List[str],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
]:
    """
    Get blendshape names, scores, and head rotation angles for a frame.

    Args:
        frames: Array of frames of shape (temporal_dim, height, width, channels)
        frame_idx: Index of the frame to get info for
        detector: FaceLandmarker model

    Returns:
        Tuple of blendshape names, frame scores, pitch, yaw, and roll angles
    """
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[frame_idx])
    detection_result = detector.detect(image)
    frame_scores = np.array(
        [blendshape.score for blendshape in detection_result.face_blendshapes[0]]
    )
    blendshape_names = [
        blendshape.category_name for blendshape in detection_result.face_blendshapes[0]
    ]
    pitch, yaw, roll = head_rotation(
        detection_result.face_landmarks[0], image.height, image.width
    )
    return blendshape_names, frame_scores, pitch, yaw, roll


def get_face_3d_model():
    """
    Returns 3D facial landmarks model points.
    These points represent key facial features in 3D space.
    """
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (225.0, 170.0, -135.0),  # Left eye left corner
            (-225.0, 170.0, -135.0),  # Right eye right corner
            (150.0, -150.0, -125.0),  # Left mouth corner
            (-150.0, -150.0, -125.0),  # Right mouth corner
        ],
        dtype=np.float64,
    )

    return model_points


def get_mediapipe_landmarks(face_landmarks, image_shape):
    """
    Extract key landmarks from MediaPipe face mesh and convert to 2D points.

    Args:
        face_landmarks: MediaPipe face landmarks
        image_shape: Tuple of (height, width) of the input image
    """
    # MediaPipe indices for the landmarks we want
    FACE_LANDMARKS = {
        "nose_tip": 4,
        "chin": 152,
        "left_eye_corner": 33,
        "right_eye_corner": 263,
        "left_mouth_corner": 61,
        "right_mouth_corner": 291,
    }

    height, width = image_shape
    image_points = []

    for landmark_name, idx in FACE_LANDMARKS.items():
        landmark = face_landmarks[idx]
        x = landmark.x * width
        y = landmark.y * height
        image_points.append([x, y])

    return np.array(image_points, dtype=np.float64)


def fix_flips(
    head_angles: NDArray[np.float32], max_angle_change: float = 90.0
) -> NDArray[np.float32]:
    """
    Fix flips in head angles by checking the distance between consecutive frames.

    Args:
        head_angles: Sequence of pitch, yaw, roll values of shape (temporal_dim, 3)
        max_angle_change: Maximum angle change allowed between frames

    Returns:
        Array of fixed head angles of shape (temporal_dim, 3)
    """
    for i in range(1, head_angles.shape[0]):
        for j in range(3):
            if abs(head_angles[i, j] - head_angles[i - 1, j]) > max_angle_change:
                head_angles[i, j] = head_angles[i - 1, j]
    return head_angles


def head_rotation(face_landmarks, height, width, camera_matrix=None, dist_coeffs=None):
    """
    Calculate head pose using OpenCV's solvePnP.

    Args:
        face_landmarks: MediaPipe face landmarks
        image_shape: Tuple of (height, width) of input image
        camera_matrix: Optional 3x3 camera intrinsic matrix
        dist_coeffs: Optional distortion coefficients

    Returns:
        tuple: (pitch, yaw, roll) angles in degrees
    """
    # If camera matrix is not provided, use a rough estimate
    if camera_matrix is None:
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )

    # If distortion coefficients are not provided, assume no distortion
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))

    # Get 3D model points and 2D image points
    model_points = get_face_3d_model()
    image_points = get_mediapipe_landmarks(face_landmarks, (height, width))

    # Solve for pose
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    # Convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Convert rotation matrix to Euler angles
    pose_mat = cv2.hconcat([rotation_mat, translation_vec])
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    # Extract angles
    pitch, yaw, roll = euler_angles.flatten()

    # rotate yaw and roll to match video
    yaw = -yaw
    roll = -roll

    return pitch, yaw, roll


def rotation_matrix_to_angles(
    rotation_matrix: NDArray[np.float64],
) -> Tuple[float, float, float]:
    """
    Calculate Euler angles from rotation matrix.
    Args:
        rotation_matrix: A 3*3 matrix with the following structure
    [Cosz*Cosy  Cosz*Siny*Sinx - Sinz*Cosx  Cosz*Siny*Cosx + Sinz*Sinx]
    [Sinz*Cosy  Sinz*Siny*Sinx + Sinz*Cosx  Sinz*Siny*Cosx - Cosz*Sinx]
    [  -Siny             CosySinx                   Cosy*Cosx         ]

    Returns:
        Tuple of pitch, yaw, and roll angles in degrees
    """
    x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    y = math.atan2(
        -rotation_matrix[2, 0],
        math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2),
    )
    z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return np.array([x, y, z]) * 180.0 / math.pi


def plot_blendshapes(frames, frame_idx, frame_scores, blendshape_names):
    fig, ax = plt.subplots(1, 2, figsize=(28, 12))
    ax[0].imshow(frames[frame_idx])
    ax[0].axis(False)

    ax[1].barh(blendshape_names, frame_scores)
    ax[1].set_ylabel("Coefficients")
    ax[1].set_xlabel("Score")
    ax[1].set_title("Blendshapes")
    return fig


def smooth_array(arr: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    """
    Smooth an array using a moving average filter.

    Args:
        arr: Array to smooth
        window: Size of the moving average window

    Returns:
        Smoothed array of the same shape as the input
    """
    window = min(window, arr.shape[0])
    kernel = np.array([1] * window) / window
    smoothed_arr = np.ones_like(arr)

    for i in range(arr.shape[1]):
        smoothed_arr[:, i] = np.convolve(arr[:, i], kernel, mode="same")
    return smoothed_arr


def get_video_landmarks(
    frames: NDArray[np.uint8],
    detector: FaceLandmarker,
    bad_frame_percent: Optional[float] = None,
) -> Tuple[List[str], NDArray[np.float32], NDArray[np.float32]]:
    """
    Get smoothed blendshape coefficients and head rotation angles for a video.

    Args:
        frames: Array of frames of shape (temporal_dim, height, width, channels)
        detector: FaceLandmarker model
    Returns:
        Tuple of blendshape names, blendshape coefficients, and head
        angles
    """
    blendshape_names = []
    bad_frames = []
    for idx, frame in enumerate(frames):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(image)
        if detection_result.face_blendshapes != []:
            blendshape_names = [
                blendshape.category_name
                for blendshape in detection_result.face_blendshapes[0]
            ]
            frame_scores = np.array(
                [
                    blendshape.score
                    for blendshape in detection_result.face_blendshapes[0]
                ]
            )
            head_angle_scores = head_rotation(
                detection_result.face_landmarks[0], image.height, image.width
            )
        else:
            bad_frames.append(idx)
            frame_scores = np.zeros((1, 52))
            head_angle_scores = np.zeros((1, 3))

        if idx == 0:
            blendshapes = frame_scores
            head_angles = head_angle_scores
        else:
            blendshapes = np.vstack((blendshapes, frame_scores))
            head_angles = np.vstack(
                (
                    head_angles,
                    head_angle_scores,
                )
            )

    if blendshape_names == []:
        raise ValueError("No face detected in the video.")
    percent_bad = len(bad_frames) / len(frames)
    if bad_frame_percent is not None and percent_bad > bad_frame_percent:
        raise ValueError(
            f"Failed to detect face in {len(bad_frames)}/{len(frames)} "
            f"frames ({percent_bad * 100}%), exceeding threshold of {bad_frame_percent}"
        )
    return blendshape_names, blendshapes, head_angles, bad_frames


def euler_to_rotation_matrix(pitch: float, yaw: float, roll: float) -> torch.Tensor:
    """
    Convert Euler angles (in degrees) to rotation matrix.
    Uses extrinsic rotation order: Yaw(Y) -> Pitch(X) -> Roll(Z)

    Args:
        pitch: Pitch angle in degrees
        yaw: Yaw angle in degrees
        roll: Roll angle in degrees

    Returns:
        Rotation matrix of shape (3, 3)
    """
    # Convert to radians
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    # Rotation matrices for each axis
    Rx = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ]
    )

    Ry = torch.tensor(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ]
    )

    Rz = torch.tensor(
        [
            [np.cos(roll), -np.sin(roll), 0.0],
            [np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Combined rotation matrix (Rz * Rx * Ry)
    R = Rz @ Rx @ Ry
    return R


def regularize_blendshapes(
    blendshapes: Union[torch.Tensor, np.ndarray], bad_frames: Optional[List[int]] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Regularize blendshapes so that minimum value of each blendshape is 0.

    Args:
        blendshapes: Blendshape coefficients of shape (B, T, D) or (T, D)
            B: batch size (optional)
            T: number of frames
            D: number of blendshape coefficients
        bad_frames: List of indices of frames where face detection failed.
            These frames will be ignored when computing minimum values.

    Returns:
        Regularized blendshape coefficients of same shape as input

    Raises:
        ValueError: If input tensor dimensions are invalid
    """
    if bad_frames is None:
        bad_frames = []

    # Validate input dimensions
    if blendshapes.ndim not in [2, 3]:
        raise ValueError(f"Expected 2D or 3D tensor, got {blendshapes.ndim}D")

    # Create copy for minimum search
    if isinstance(blendshapes, np.ndarray):
        min_search_array = np.copy(blendshapes)

        # Handle bad frames
        if min_search_array.ndim == 3:
            min_search_array[:, bad_frames, :] = np.inf
        else:
            min_search_array[bad_frames, :] = np.inf

        # Compute minimums along time dimension
        min_vals = min_search_array.min(axis=-2, keepdims=True)

    else:  # torch.Tensor
        min_search_array = blendshapes.clone()

        # Handle bad frames
        if min_search_array.ndim == 3:
            min_search_array[:, bad_frames, :] = float("inf")
        else:
            min_search_array[bad_frames, :] = float("inf")

        # Compute minimums along time dimension
        min_vals = min_search_array.min(dim=-2, keepdim=True).values

    # Subtract minimum values to regularize, except from bad frames
    good_frames = [i for i in range(blendshapes.shape[-2]) if i not in bad_frames]
    regularized = (
        blendshapes.clone()
        if isinstance(blendshapes, torch.Tensor)
        else np.copy(blendshapes)
    )
    regularized[..., good_frames, :] = blendshapes[..., good_frames, :] - min_vals
    return regularized


def clean_up_blendshapes(
    blendshapes: np.ndarray | torch.Tensor,
    mouth_exaggeration: float,
    brow_exaggeration: float,
    unsquinch_fix: float,
    eye_contact_fix: float,
    clear_neutral: bool = False,
    exaggerate_above: float = 0,
    symmetrize_eyes: bool = False,
) -> np.ndarray:
    """
    Exaggerate blendshapes by a given factor.

    Args:
        blendshapes: Blendshape coefficients of shape (B, T, D) or (T, D)
        mouth_exaggeration: Factor to exaggerate mouth blendshapes by
        brow_exaggeration: Factor to exaggerate brow blendshapes by
        unsquinch_fix: Amount to fix squinting blendshapes by
        eye_contact_fix: Amount to fix eye contact blendshapes by
        clear_neutral: Whether to clear the neutral blendshape
        exaggerate_above: Minimum value to exaggerate blendshapes above
        symmetrize_eyes: Whether to symmetrize eye blendshapes

    Returns:
        Exaggerated blendshape coefficients of shape (B, T, D) or (T, D)
    """

    def modify_blendshapes(
        blendshapes: np.ndarray, target_substrings: List[str], factor: float
    ) -> np.ndarray:
        if factor != 1:
            for i, shape in enumerate(BLENDSHAPE_NAMES):
                if any(substring in shape for substring in target_substrings):
                    blendshapes_offset = blendshapes[..., i] - exaggerate_above
                    blendshapes[..., i] = blendshapes_offset * factor + exaggerate_above
        if isinstance(blendshapes, torch.Tensor):
            blendshapes = torch.clamp(blendshapes, 0.0, 1.0)
        else:
            blendshapes = np.clip(blendshapes, 0.0, 1.0)
        return blendshapes

    if clear_neutral:
        blendshapes[..., 0] = 0

    modify_blendshapes(blendshapes, ["mouth", "jaw", "cheek"], mouth_exaggeration)
    modify_blendshapes(blendshapes, ["brow", "noseSneer", "eye"], brow_exaggeration)
    if unsquinch_fix > 0:
        eye_idx = [
            i
            for i, name in enumerate(BLENDSHAPE_NAMES)
            if "eyeSquint" in name or "eyeBlink" in name
        ]
        for idx in eye_idx:
            blendshapes[..., idx] -= unsquinch_fix
    if eye_contact_fix > 0:
        eye_idx = [i for i, name in enumerate(BLENDSHAPE_NAMES) if "eyeLook" in name]
        for idx in eye_idx:
            blendshapes[..., idx] -= eye_contact_fix
    if symmetrize_eyes:
        # average between eyeBlinkLeft and eyeBlinkRight
        eye_blink_left_index = BLENDSHAPE_NAMES.index("eyeBlinkLeft")
        eye_blink_right_index = BLENDSHAPE_NAMES.index("eyeBlinkRight")
        avg_val = (
            blendshapes[..., eye_blink_left_index]
            + blendshapes[..., eye_blink_right_index]
        ) / 2
        blendshapes[..., eye_blink_left_index] = avg_val
        blendshapes[..., eye_blink_right_index] = avg_val

    if isinstance(blendshapes, torch.Tensor):
        blendshapes = torch.clamp(blendshapes, 0.0, 1.0)
    else:
        blendshapes = np.clip(blendshapes, 0.0, 1.0)

    return blendshapes


def exaggerate_head_wiggle(
    head_angles: np.ndarray[np.float32] | torch.Tensor, exaggeration_factor: float
) -> np.ndarray[np.float32]:
    """
    Exaggerate head angles by a given factor.

    Args:
        head_angles: Sequence of pitch, yaw, roll values of shape (temporal_dim, 3)
        exaggeration_factor: Factor to exaggerate the head angles by

    Returns:
        Exaggerated head angles of shape (temporal_dim, 3)
    """
    return head_angles * exaggeration_factor


def deblink(landmark_array: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Removes blinks from the landmark array by interpolating the missing values.

    Args:
        landmark_array: Array of shape [time_dim, 52] containing blendshape values

    Returns:
        Array with same shape as input but with blink events interpolated
    """
    eyeBlink_index = [
        i for i, name in enumerate(BLENDSHAPE_NAMES) if "eyeBlink" in name
    ]

    # Create copy to avoid modifying input
    output = landmark_array.copy()

    # Get average blink values across both eyes
    blink_signal = np.mean(landmark_array[:, eyeBlink_index], axis=1)

    # Detect blinks using threshold
    BLINK_THRESHOLD = 0.1
    blink_mask = blink_signal > BLINK_THRESHOLD

    # Find contiguous blink regions
    from scipy.ndimage import label

    blink_regions, num_regions = label(blink_mask)

    # Interpolate over each blink
    for region in range(1, num_regions + 1):
        blink_indices = np.where(blink_regions == region)[0]
        start_idx = blink_indices[0] - 1
        end_idx = blink_indices[-1] + 1

        # Handle edge cases
        if start_idx < 0:
            start_idx = 0
        if end_idx >= len(landmark_array):
            end_idx = len(landmark_array) - 1

        # Interpolate all features during blink
        for i in range(landmark_array.shape[1]):
            output[blink_indices, i] = np.interp(
                blink_indices,
                [start_idx, end_idx],
                [landmark_array[start_idx, i], landmark_array[end_idx, i]],
            )

    return output
