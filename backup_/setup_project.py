"""
setup_project.py
프로젝트 기본 구조 자동 생성 스크립트
"""

import os
from pathlib import Path

# 폴더 트리 정의
# PROJECT_NAME은 원하는 프로젝트 이름으로 변경 가능
PROJECT_NAME = "LoRA_Style_Classifier"

FOLDERS = [
    "data/raw",
    "data/processed",
    "src",
    "scripts",
    "app",
    "notebooks",
    "outputs/models",
    "outputs/logs",
    "outputs/results",
]

FILES = {
    "README.md": f"# {PROJECT_NAME}\n\nLoRA Style Classifier 프로젝트 초기 설정 완료.\n\n구성 폴더:\n- data/: 원본 및 전처리 데이터\n- src/: 학습 코드 및 모듈\n- app/: UI(Web) 코드\n- scripts/: 실행 스크립트\n\n다음 단계 → A-2: src 코드 프로토타입 작성",
    "requirements.txt": "torch\ntorchvision\npandas\nnumpy\nmatplotlib\nseaborn\nscikit-learn\ntqdm\nPillow\nstreamlit\ngradio\nrequests",
    ".gitignore": "__pycache__/\noutputs/\ndata/\n*.pth\n*.log\n.ipynb_checkpoints/\n.env\n",
}

SRC_FILES = [
    "src/config.py",
    "src/model.py",
    "src/train.py",
    "src/data_loader.py",
    "src/preprocess.py",
    "src/evaluate.py",
    "src/visualize.py",
    "src/utils.py",
]

SCRIPT_FILES = [
    "scripts/download_civitai_data.py",
    "scripts/split_dataset.py",
    "scripts/run_training.py",
    "scripts/run_inference.py",
    "scripts/export_model.py",
]

NOTEBOOK_FILES = [
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_training_experiment.ipynb",
    "notebooks/03_evaluation_report.ipynb",
]

APP_FILES = [
    "app/app.py",
    "app/requirements.txt",
]


# 디렉토리 생성 함수
def create_structure():
    root = Path(PROJECT_NAME)
    root.mkdir(exist_ok=True)

    for folder in FOLDERS:
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)

    for filepath, content in FILES.items():
        fpath = root / filepath
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)

    for filepath in SRC_FILES + SCRIPT_FILES + NOTEBOOK_FILES + APP_FILES:
        fpath = root / filepath
        if not fpath.exists():
            fpath.touch()

    print(f"✅ '{PROJECT_NAME}' 폴더 구조가 생성되었습니다.")
    print("📁 생성된 주요 폴더:")
    for folder in FOLDERS:
        print("  └──", folder)
    print("\n다음 단계 → A-2: src 폴더의 코드 프로토타입 작성")


# 실행
if __name__ == "__main__":
    create_structure()
