"""
mock_drone_server.py
=====================
Mimics the real drone server (168.107.63.33:5001) locally.
Streams sample.mp4 as a never-ending MJPEG stream so the analyzer
can be tested against a live-like source self-contained.
 
Endpoints:
  GET /video/{drone_id}   - MJPEG stream (multipart/x-mixed-replace)
  GET /health             - status
  GET /drones             - active drone list
 
Run:
  python mock_drone_server.py
  -> http://localhost:9999/video/test
"""
 
import time
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
 
VIDEO_PATH = '/workspace/guseok/sample.mp4'
FPS        = 30
PORT       = 9999
 
app = FastAPI(title="Mock Drone Server")
 
 
def gen_frames():
    """Read sample.mp4 in an infinite loop, yielding JPEG frames."""
    while True:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[mock] cannot open video file: {VIDEO_PATH}")
            time.sleep(1)
            continue
 
        n = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +
                    jpeg.tobytes() +
                    b'\r\n'
                )
                n += 1
            # time.sleep(1.0 / FPS)
 
        cap.release()
        print(f"[mock] one loop done (frames={n}), restarting")
 
 
@app.get("/video/{drone_id}")
def video(drone_id: str):
    print(f"[mock] client connected: drone_id={drone_id}")
    return StreamingResponse(
        gen_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame',
    )
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.get("/drones")
def drones():
    return {"drones": ["test"]}
 
 
if __name__ == "__main__":
    import uvicorn
    print(f"[mock] starting on port {PORT}, video={VIDEO_PATH}")
    print(f"[mock] analyzer URL: http://localhost:{PORT}/video/test")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
