# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
import copy
import json
import subprocess
from typing import NamedTuple, Optional

import cv2
import ffmpeg
import numpy as np

# openpose setup
from src import util
from src.body import Body
from src.hand import Hand


class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        file_path,
    ]
    result = subprocess.run(
        command_array,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return FFProbeResult(
        return_code=result.returncode, json=result.stdout, error=result.stderr
    )


body_estimation = Body("model/body_pose_model.pth")
hand_estimation = Hand("model/hand_pose_model.pth")


def process_frame(frame, body=True, hands=True):
    canvas = copy.deepcopy(frame)
    if body:
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)
    if hands:
        hands_list = util.handDetect(candidate, subset, frame)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(frame[y : y + w, x : x + w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            all_hand_peaks.append(peaks)
        canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas


# open specified video
parser = argparse.ArgumentParser(
    description="Process a video annotating poses detected."
)
parser.add_argument("file", type=str, help="Video file location to process.")
parser.add_argument("--no_hands", action="store_true", help="No hand pose")
parser.add_argument("--no_body", action="store_true", help="No body pose")
args = parser.parse_args()
video_file = args.file
cap = cv2.VideoCapture(video_file)

# get video file info
ffprobe_result = ffprobe(args.file)
info = json.loads(ffprobe_result.json)
videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
input_fps = videoinfo["avg_frame_rate"]
# Calculate numerical fps value from fraction format like "30000/1001"
if "/" in input_fps:
    num, den = input_fps.split("/")
    input_fps = float(num) / float(den)
else:
    input_fps = float(input_fps)
input_pix_fmt = videoinfo["pix_fmt"]
input_vcodec = videoinfo["codec_name"]

# define a writer object to write to a movidified file
postfix = info["format"]["format_name"].split(",")[0]
output_file = ".".join(video_file.split(".")[:-1]) + ".processed." + postfix

# Ensure width is even for H.264 encoding
width = int(videoinfo["width"])
height = int(videoinfo["height"])
if width % 2 == 1:
    width -= 1

command_array = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel",
    "error",
    "-f",
    "rawvideo",
    "-vcodec",
    "rawvideo",
    "-pixel_format",
    "bgr24",
    "-video_size",
    f"{width}x{height}",
    "-framerate",
    str(input_fps),
    "-i",
    "-",
    "-vf",
    f"crop={width}:{height}:0:0",
    "-c:v",
    "libx264",
    "-preset",
    "medium",
    "-f",
    "mp4",
    output_file,
]


class Writer:
    def __init__(
        self, output_file, input_fps, input_framesize, input_pix_fmt, input_vcodec
    ):
        # Ensure width is even for H.264 encoding
        width = input_framesize[1]
        if width % 2 == 1:
            width -= 1
        height = input_framesize[0]

        self.ff_proc = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s="%sx%s" % (width, height),
                r=input_fps,
            )
            .output(output_file, pix_fmt=input_pix_fmt, vcodec=input_vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def __call__(self, frame):
        # If width is odd, crop it
        if frame.shape[1] % 2 == 1:
            frame = frame[:, :-1]
        self.ff_proc.stdin.write(frame.tobytes())

    def close(self):
        self.ff_proc.stdin.close()
        self.ff_proc.wait()


writer: Optional[Writer] = None
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    posed_frame = process_frame(frame, body=not args.no_body, hands=not args.no_hands)

    if writer is None:
        input_framesize = posed_frame.shape[:2]
        writer = Writer(
            output_file, input_fps, input_framesize, input_pix_fmt, input_vcodec
        )

    cv2.imshow("frame", posed_frame)

    # write the frame
    writer(posed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
if writer is not None:
    writer.close()
cv2.destroyAllWindows()
