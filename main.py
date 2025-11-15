from fastapi import FastAPI, UploadFile, File, HTTPException
import io
from PIL import Image, ImageOps # 👈 ImageOps 추가 (회전 처리용)
from ultralytics import YOLO

# FastAPI 앱 생성
app = FastAPI()

# --- 💡 모델 로드 부분 💡 ---
try:
    model = YOLO("best.pt")
    print("YOLOv8 모델 로드 성공: best.pt")
    print(f"모델 클래스: {model.names}") # 👈 [추가] 모델이 인식하는 클래스 이름들 출력
except Exception as e:
    print(f"YOLOv8 모델 로드 실패: {e}")
    model = None

# 1. 기본 접속 테스트용
@app.get("/")
def read_root():
    return {"message": "헬멧 감지 AI 서버에 오신 것을 환영합니다!"}

# 2. 이미지 업로드 및 예측 엔드포인트
@app.post("/predict")
async def predict_helmet(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="AI 모델이 로드되지 않았습니다.")

    # 1. 받은 이미지 파일 읽기
    file_bytes = await file.read()
    
    # 2. byte 데이터를 Pillow 이미지 객체로 변환
    try:
        image = Image.open(io.BytesIO(file_bytes))
        
        # 👇 [수정 1] 핸드폰 사진 회전 문제(EXIF) 자동 보정
        image = ImageOps.exif_transpose(image)
        
        image = image.convert("RGB")
        
        # 👇 [디버깅용] 서버에 수신된 이미지를 저장해서 눈으로 확인
        image.save("debug_received_image.jpg")
        print("📸 수신된 이미지 저장 완료: debug_received_image.jpg")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 파일 처리 오류: {e}")
    
    # --- 💡 모델 추론 로직 (수정됨) 💡 ---
    try:
        # 수정 후 (학습할 때 640으로 했다면, 추론도 640으로!)
        results = model(image, conf=0.25, imgsz=640, save=True)
        # (save=True: 'runs/detect/predict' 폴더에 결과 이미지가 저장됨)
        
    except Exception as e:
        print(f"❌ AI 추론 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=f"AI 추론 오류: {e}")

    # 결과 파싱 및 가공
    detection_results = []
    
    # 👇 [디버깅용] 탐지된 객체 수 출력
    if results:
        print(f"🔍 탐지된 객체 수: {len(results[0].boxes)}")
    else:
        print("🔍 탐지된 객체 없음.")

    for r in results:
        if r.boxes: 
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [float(coord) for coord in box.xyxy[0]]
                
                class_name = r.names.get(class_id, "unknown") 

                # 👇 [디버깅용] 탐지된 모든 객체와 점수 출력
                print(f"  -> 찾음: {class_name}, 점수: {confidence:.2f}")

                detection_results.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

    # 3. 최종 결과 반환
    return {
        "filename": file.filename,
        "file_size": len(file_bytes),
        "detections": detection_results, # 탐지된 객체들의 리스트
        "message": "헬멧 감지 추론 완료"
    }