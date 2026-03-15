# Directory Structure (Clean Version)

สรุปโครงสร้างโปรเจกต์หลังจากทำ Refactor เพื่อความเป็นระเบียบ (Clean Architecture):

### 📁 Root Directories
- `agent_docs/`: แหล่งเก็บเอกสารระสถาปัตยกรรมและโครงสร้างโปรเจกต์
- `artifacts/`: เก็บไฟล์ผลลัพธ์ถาวรที่ได้จากการเทรนโมเดล
    - `models/`: ไฟล์โมเดล `.pkl` (ย้ายมาจาก `save_models/`)
    - `scalers/`: ไฟล์ตัวแปลงข้อมูล `.pkl` (ย้ายมาจาก `save_scalers/`)
    - `metadata/`: (Reserved) สำหรับเก็บ Metadata เฉพาะของแต่ละ Version
    - `local_cache.duckdb`: ไฟล์ Cache สำหรับใช้งาน DuckDB (พักข้อมูลจาก Postgres บน Native)
- `outputs/`: เก็บผลลัพธ์ที่เปลี่ยนตามการรัน (Dynamic Outputs)
    - `data/`: เก็บไฟล์ CSV, Metrics และ Metadata ของการรันล่าสุด (ย้ายมาจาก `output_data/`)
    - `plots/`: เก็บกราฟและรูปภาพต่างๆ (ย้ายมาจาก `plots/`)
- `backups/`: เก็บไฟล์สำรองข้อมูลหรือไฟล์เก่า (เช่น `Dockerfile_backup`)
- `app/`: Source code หลักของแอปพลิเคชัน
    - `data/`: Logic การจัดการข้อมูล (Loading, Cleaning, Engineering)
    - `models/`: นิยามของโมเดลและการเทรน (Lucis, Hamilton DAGs เช่น `features_hamilton.py` และ `training_hamilton.py`)
    - `prediction/`: Logic สำหรับการทำ Inference/Prediction
    - `utils/`: รวมไฟล์ Helper ต่างๆ เช่น `utils_model.py`, `utils_mlflow.py` และ `duckdb_utils.py`
    - `config/`: ไฟล์ตั้งค่าต่างๆ ของระบบ
    - `routes.py`: API Endpoints ทั้งหมด
- `dagster_project/`: Orchestration code สำหรับควบคุม Pipeline
- `dbt_project/`: SQL transformations (ELT layer)
- `feature_repo/`: Feast feature store definitions และ metadata
- `mlruns/`: ข้อมูลการทดลองของ MLflow

### 📄 Key Files
- `README.md`: รายละเอียดการติดตั้งและการใช้งานเบื้องต้น
- `requirements.txt`: รายการ Dependencies ของโปรเจกต์
- `Dockerfile`: สำหรับการทำ Containerization
- `.env`: เก็บ Environment Variables (Private)
