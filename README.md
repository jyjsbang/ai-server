### 📋 README.md
````markdown
# ⛑️ Helmet Detection AI Server

**FastAPI**와 **YOLOv8**을 기반으로 개발된 실시간 헬멧 착용 감지 AI 서버입니다.  
클라이언트(React Native 등)에서 전송된 이미지를 분석하여 헬멧 착용 여부, 객체 위치(BBox), 신뢰도(Confidence) 정보를 JSON으로 반환합니다.

## 🚀 Key Features

* **FastAPI 기반 고성능 서버:** 비동기(`async`) 처리를 통해 빠른 추론 응답 속도를 제공합니다.
* **YOLOv8 Object Detection:** 커스텀 데이터셋(Roboflow, 약 8,000장)으로 학습된 모델을 사용하여 높은 정확도를 보장합니다.
* **자동 이미지 보정 (Auto-Rotation):** 모바일 기기에서 촬영 시 발생하는 이미지 회전(EXIF) 문제를 `PIL.ImageOps`를 통해 자동으로 보정합니다.
* **RESTful API:** 간편한 연동을 위한 직관적인 `/predict` 엔드포인트를 제공합니다.
* **Debug Mode:** 서버 실행 시 수신된 원본 이미지와 추론 결과 이미지를 로컬에 저장하여 디버깅이 용이합니다.

## 🛠 Tech Stack

| Category | Technology | Version (Key) |
| :--- | :--- | :--- |
| **Language** | Python | 3.10+ |
| **Framework** | FastAPI | 0.121.1 |
| **Server** | Uvicorn | 0.38.0 |
| **AI Model** | Ultralytics YOLOv8 | 8.3.228 |
| **Image Proc** | Pillow (PIL) | 12.0.0 |
| **ML Core** | Torch | 2.9.1 |

*(상세 의존성 버전은 `requirements.txt` 참조)*

---

## 💾 Installation & Setup

이 프로젝트를 로컬 환경에서 실행하기 위한 단계별 가이드입니다.

### 1. 레포지토리 클론 (Clone)
```bash
git clone [https://github.com/jyjsbang/ai-server.git](https://github.com/jyjsbang/ai-server.git)
cd ai-server
````

### 2\. 가상환경 생성 및 활성화

**Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3\. 라이브러리 설치

`requirements.txt`에 명시된 버전으로 설치합니다.

```bash
pip install -r requirements.txt
```

### 4\. 모델 파일(Weights) 준비 ⚠️ 필수

GitHub 용량 제한으로 인해 학습된 모델 파일은 포함되어 있지 않습니다.

  * 프로젝트 루트 디렉토리에 **`best.pt`** 파일이 있어야 합니다.
  * (팀원에게 공유받은 `best.pt` 파일을 `main.py`와 같은 위치에 넣어주세요.)

-----

## 🏃‍♂️ How to Run

서버를 실행하면 기본적으로 `http://127.0.0.1:8000` 포트에서 작동합니다.

```bash
uvicorn main:app --reload
```

-----

## 📡 API Documentation

### 1\. Health Check

서버 상태를 확인합니다.

  * **URL:** `GET /`
  * **Response:**
    ```json
    {
        "message": "헬멧 감지 AI 서버에 오신 것을 환영합니다!"
    }
    ```

### 2\. Predict Helmet (헬멧 감지 요청)

이미지를 전송하여 헬멧 착용 여부를 분석합니다.

  * **URL:** `POST /predict`
  * **Header:** `Content-Type: multipart/form-data`
  * **Body:** `file` (Binary Image Data)
  * **Parameters:**
      * `conf`: 0.25 (Default confidence threshold)
      * `imgsz`: 640 (Inference image size)
  * **Response Example:**
    ```json
    {
        "filename": "image.jpg",
        "file_size": 102400,
        "detections": [
            {
                "class_name": "helmet",
                "confidence": 0.88,
                "bbox": {
                    "x1": 120.5,
                    "y1": 50.0,
                    "x2": 300.2,
                    "y2": 200.8
                }
            }
        ],
        "message": "헬멧 감지 추론 완료"
    }
    ```

-----

## 📂 Project Structure

```bash
ai-server/
├── venv/                   # 가상환경 (Git 제외)
├── runs/                   # YOLO 추론 결과 저장 폴더 (자동 생성)
├── best.pt                 # 학습된 YOLO 모델 가중치 파일 (Git 제외)
├── debug_received_image.jpg # 디버깅용 수신 이미지 (자동 생성)
├── main.py                 # FastAPI 메인 서버 코드
├── requirements.txt        # 의존성 라이브러리 목록
└── README.md               # 프로젝트 설명서
```

## ⚠️ Notes for Developers

  * **디버깅 파일:** 서버가 실행되면 현재 폴더에 `debug_received_image.jpg`와 `runs/detect/predict` 폴더가 생성되어 이미지가 계속 저장됩니다. 배포(Production) 단계에서는 `main.py` 내의 `save=True` 옵션과 이미지 저장 코드를 주석 처리하는 것을 권장합니다.
  * **모델 버전:** 현재 코드는 `best.pt`를 로드하도록 설정되어 있습니다. 모델 파일명이 다르다면 `main.py` 10번째 줄을 수정하세요.

<!-- end list -->

```

---

### 💡 작성자가 확인해야 할 사항 (업로드 파일 기반)

1.  [cite_start]**`requirements.txt` 버전:** 업로드해주신 파일 [cite: 1]에 `torch==2.9.1`, `fastapi==0.121.1` 등 매우 최신(혹은 특정 환경의) 버전들이 명시되어 있습니다. 이 버전 정보를 README의 **Tech Stack** 섹션에 정확히 반영했습니다.
2.  **`debug_received_image.jpg`:** `main.py`를 보면 이미지를 받을 때마다 `debug_received_image.jpg`로 저장하고 있습니다. 이 부분은 개발 편의를 위한 것이므로 README의 **Notes for Developers** 섹션에 안내 문구를 추가했습니다.
```
