import cv2
import time
import numpy as np
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 576)},controls={"AfMode": 2}))   # RGB888
picam2.start()
picam2.set_controls({"AfTrigger": 0})  # inicia foco automático

cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

fps = 0
t0 = time.time()

try:
    while True:
        frame = picam2.capture_array("main")  # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        fps = 1.0 / (time.time() - t0)
        t0 = time.time()

        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Preview", frame_bgr)
        print(f"FPS: {fps:.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
