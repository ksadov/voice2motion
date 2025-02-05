import cv2
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
from pathlib import Path


def combine_videos_side_by_side(
    reference_video_path: Path, generated_video_path: Path, output_path: Path
) -> None:
    """
    Combines two videos side by side and adds audio from the reference video.
    The output video length will match the shorter of the two input videos.

    Args:
        reference_video_path: Path to the reference video with audio
        generated_video_path: Path to the generated video
        output_path: Path where the combined video will be saved
    """
    # Convert paths to strings for cv2 and ensure they exist
    if not reference_video_path.exists():
        raise FileNotFoundError(f"Reference video not found: {reference_video_path}")
    if not generated_video_path.exists():
        raise FileNotFoundError(f"Generated video not found: {generated_video_path}")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open both videos
    gen_cap = cv2.VideoCapture(str(generated_video_path))
    ref_cap = cv2.VideoCapture(str(reference_video_path))

    # Get video properties
    gen_width = int(gen_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gen_height = int(gen_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get frame counts
    gen_frame_count = int(gen_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_frame_count = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use the shorter video's frame count
    total_frames = min(gen_frame_count, ref_frame_count)

    # Calculate dimensions for the combined video
    # Scale reference video to match generated video height
    scale_factor = gen_height / ref_height
    new_ref_width = int(ref_width * scale_factor)
    new_ref_height = gen_height

    # Combined video dimensions
    combined_width = gen_width + new_ref_width
    combined_height = gen_height

    # Get the frame rate of the generated video
    fps = int(gen_cap.get(cv2.CAP_PROP_FPS))

    # Create temporary output path
    temp_output_path = output_path.with_name(
        f"{output_path.stem}_temp{output_path.suffix}"
    )

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(temp_output_path), fourcc, fps, (combined_width, combined_height)
    )

    # Process frames up to the shorter video's length
    for _ in range(total_frames):
        gen_ret, gen_frame = gen_cap.read()
        ref_ret, ref_frame = ref_cap.read()

        if not gen_ret or not ref_ret:
            break

        # Resize reference frame
        ref_frame_resized = cv2.resize(ref_frame, (new_ref_width, new_ref_height))

        # Combine frames
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        combined_frame[:, :gen_width] = gen_frame
        combined_frame[:, gen_width:] = ref_frame_resized

        # Write frame
        out.write(combined_frame)

    # Release resources
    gen_cap.release()
    ref_cap.release()
    out.release()

    # Add audio from reference video
    # Using moviepy to combine the video with audio
    video_clip = VideoFileClip(str(temp_output_path))
    reference_clip = VideoFileClip(str(reference_video_path))

    # Trim audio to match the shorter video length
    audio_duration = total_frames / fps
    audio_clip = reference_clip.audio.subclip(0, audio_duration)

    # Combine video with audio and write final output
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

    # Clean up temporary file and clips
    temp_output_path.unlink()
    video_clip.close()
    reference_clip.close()

    print(f"Combined video saved to {output_path}")
    print(f"Video length: {audio_duration:.2f} seconds ({total_frames} frames)")


def add_audio_to_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """
    Adds audio from an audio file to a video file.
    Args:
        video_path: Path to the video file
        audio_path: Path to the audio file
        output_path: Path to save the output video
    """
    # Check if input files exist
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_clip = VideoFileClip(str(video_path))
    audio_clip = AudioFileClip(str(audio_path))

    # Get durations
    video_duration = video_clip.duration
    audio_duration = audio_clip.duration

    # Use the shorter duration of the two
    final_duration = min(video_duration, audio_duration)

    # Trim both video and audio to the shorter duration
    video_clip = video_clip.subclip(0, final_duration)
    audio_clip = audio_clip.subclip(0, final_duration)

    # Combine video with audio and write output
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

    # Clean up
    video_clip.close()
    audio_clip.close()
    final_clip.close()

    print(f"Audio added to video and saved to {output_path}")
    print(f"Final duration: {final_duration:.2f} seconds")
    if video_duration != audio_duration:
        print(f"Note: Original video duration was {video_duration:.2f} seconds")
        print(f"      Original audio duration was {audio_duration:.2f} seconds")
        print(f"      Output was trimmed to match shorter duration")
