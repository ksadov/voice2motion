import argparse
import os
import sys
import subprocess
from multiprocessing.pool import ThreadPool
from yt_dlp import YoutubeDL
from tqdm import tqdm


class VidInfo:
    def __init__(self, yt_id, start_time, end_time, outdir, video_format="best"):
        self.yt_id = yt_id
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.outdir = os.path.join(outdir, str(yt_id))
        self.out_filename = os.path.join(
            self.outdir, f"{yt_id}_{start_time}_{end_time}_av.mp4"
        )
        self.video_format = video_format


def download(vidinfo):
    if os.path.exists(vidinfo.out_filename):
        return f"{vidinfo.yt_id}, SKIPPED (already exists)!"

    yt_base_url = "https://www.youtube.com/watch?v="
    yt_url = yt_base_url + vidinfo.yt_id

    ydl_opts = {
        "format": vidinfo.video_format,
        "quiet": True,
        "ignoreerrors": True,
        "no_warnings": True,
        "sleep_interval": 2,
        "max_sleep_interval": 4,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url=yt_url, download=False)
            if not info or "url" not in info:
                return f"{vidinfo.yt_id}, ERROR (youtube - no URL found)!"
            video_url = info["url"]
    except Exception:
        return f"{vidinfo.yt_id}, ERROR (youtube)!"

    try:
        os.makedirs(vidinfo.outdir, exist_ok=True)

        subprocess.run(
            [
                "ffmpeg",
                "-ss",
                str(vidinfo.start_time),
                "-to",
                str(vidinfo.end_time),
                "-i",
                video_url,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-b:a",
                "192k",
                "-y",
                vidinfo.out_filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

    except subprocess.CalledProcessError:
        # If ffmpeg fails, remove the empty directory we just created
        try:
            os.rmdir(vidinfo.outdir)
        except OSError:
            pass  # Directory might not be empty
        return f"{vidinfo.yt_id}, ERROR (ffmpeg)!"

    return f"{vidinfo.yt_id}, DONE!"


def main():
    parser = argparse.ArgumentParser(
        description="Download videos from YouTube and clip them to a single MP4 file containing both video and audio."
    )
    parser.add_argument(
        "--csv",
        default="assets/avspeech_test.csv",
        help="CSV file path (default: assets/avspeech_test.csv).",
    )
    parser.add_argument(
        "--outdir",
        default="data/avspeech_test",
        help="Target output directory (default: data/avspeech_test).",
    )
    parser.add_argument(
        "--format", default="best", help="Video format to extract (default: best)."
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of CSV entries to download (default: no limit).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files already exist.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.csv, "r") as f:
        lines = [x.strip().split(",") for x in f.readlines()]

    if args.max_files is not None:
        lines = lines[: args.max_files]

    vidinfos = []
    for x in lines:
        yt_id, start_time, end_time = x[:3]
        vi = VidInfo(yt_id, start_time, end_time, args.outdir, args.format)
        vidinfos.append(vi)

    bad_files_path = os.path.join(args.outdir, "bad_files_csv.txt")
    bad_files = open(bad_files_path, "w")

    results = ThreadPool(10).imap_unordered(download, vidinfos)

    err_cnt = 0
    skip_cnt = 0
    for r in tqdm(results, total=len(vidinfos), desc="Downloading videos"):
        if "ERROR" in r:
            bad_files.write(r + "\n")
            err_cnt += 1
        elif "SKIPPED" in r:
            skip_cnt += 1

    bad_files.close()
    print(f"Total Error: {err_cnt}")
    print(f"Total Skipped: {skip_cnt}")


if __name__ == "__main__":
    main()
