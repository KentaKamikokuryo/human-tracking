import sys
import os
import time
import cv2
import torch

import copy
from tracker.bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
from fast_reid.fast_reid_interface import FastReIDInterface, FastReIDModelType
from box_detection.box_detection import BoxDetectionMMDet, BoxModelType
from utils.color import Color

def is_parsable_to_int(s):
  try:
    int(s)
    return True
  except ValueError:
    return False

disable_video_writer = False
video_name = "video4"
video = os.path.join(os.path.dirname(__file__), ".videos/" + video_name + ".mp4")

# Set the path to the pretrained model
path_model = os.path.join(os.path.dirname(__file__), "pretrained")

# Set the path to the config and checkpoint files for the box detection model
print("Setting up box model")
box_model_type = BoxModelType(path_model, BoxModelType.YOLOX)
box_config_file = box_model_type.config_file
box_checkpoint_file = box_model_type.checkpoint_file
box_detection = BoxDetectionMMDet(box_config_file, box_checkpoint_file, threshold=0.9)

# Set the path to the config and checkpoint files for the re-identification model
print("Setting up reid model")
fast_reid_model_type = FastReIDModelType(path_model, FastReIDModelType.DukeMTMC)
reid_config_file = fast_reid_model_type.config_file
reid_checkpoint_file = fast_reid_model_type.checkpoint_file
body_feature_extractor = FastReIDInterface(reid_config_file, reid_checkpoint_file, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=8)

# Model initialization
print("Initializing models...")
bot_sort = BoTSORT(box_detection, body_feature_extractor, frame_rate=60)

def main():
  # cv2.VideoCapture object
  cap = cv2.VideoCapture(int(video) if is_parsable_to_int(video) else video)
  if not disable_video_writer:
      cap_fps = cap.get(cv2.CAP_PROP_FPS)
      w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      
      fourcc = cv2.VideoWriter_fourcc(*"mp4v")
      video_writer = cv2.VideoWriter(
        ".output/" + video_name + ".mp4", 
        fourcc, 
        cap_fps, 
        (w, h)
      )
  else:
      video_writer = None

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      
      debug_image = copy.deepcopy(frame)
      # debug_image = cv2.resize(debug_image, (debug_image.shape[1] // 2, debug_image.shape[0] // 2))
      debug_image_h = debug_image.shape[0]
      debug_image_w = debug_image.shape[1]
      
      start_time = time.perf_counter()
      stracks = bot_sort.update(img=debug_image)
      elapsed_time = time.perf_counter() - start_time
      
      cv2.putText(debug_image, f"Time: {elapsed_time*1000:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
      cv2.putText(debug_image, f"Time: {elapsed_time*1000:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (  0,   0,   0), 1, cv2.LINE_AA)
      
      for strack in stracks:
        
        color = (255, 0, 0)
        cv2.rectangle(debug_image, (int(strack.tlbr[0]), int(strack.tlbr[1])), (int(strack.tlbr[2]), int(strack.tlbr[3])), (255,255,255), 2)
        cv2.rectangle(debug_image, (int(strack.tlbr[0]), int(strack.tlbr[1])), (int(strack.tlbr[2]), int(strack.tlbr[3])), color, 1)
        ptx = int(strack.tlbr[0]) if int(strack.tlbr[0])+50 < debug_image_w else debug_image_w-50
        pty = int(strack.tlbr[1])-10 if int(strack.tlbr[1])-25 > 0 else 20
        cv2.putText(debug_image, f'{strack.track_id}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug_image, f'{strack.track_id}', (ptx, pty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (  0,   0, 255), 1, cv2.LINE_AA)
        
      key = cv2.waitKey(1)
      if key == 27: # ESC
          break
      
      cv2.imshow("test", debug_image.astype("uint8"))
      if video_writer is not None:
          video_writer.write(debug_image)
          
  if video_writer is not None:
    video_writer.release() 
    print(f"Output video saved to .output/{video_name}.mp4")

  if cap is not None:
    cap.release()
    print("Capture released")
  

if __name__ == "__main__":
  main()
  print("Done")