import analyzer
analyzer.load_models()

# 영상 소스: 아무 mp4 파일 경로나 드론 스트림 URL
VIDEO_SOURCE = "http://localhost:9999/video/test"   # ← 실제 영상 파일 경로로 바꿔주세요
# 또는 드론 스트림: "http://168.107.63.33:5001/video/DRONE_ID"

results = analyzer.run_analysis(
    search_id=999,
    video_source=VIDEO_SOURCE,
    mission_data={"gender": "female", "up_color": "white"},
    query_photo_path=None,   # 사진 없이도 됨 (Re-ID 끄고 YOLO+FEM만)
    is_drone=True,
)
print("\n=== RESULTS ===")
for r in results:
    print(r)
