---

# 🧠 atto-model  
*Requires Python 3.11*

---

## 1. Python Environment Setup

### macOS / Linux
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Windows
```powershell
python -m venv venv
venv\Scripts\activate
```

---

## 2. Install Dependencies

1. Upgrade packaging tools:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```
2. Install core requirements:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Running the Web App

### macOS / Linux
```bash
python3.11 -m app.app
```

### Windows
```powershell
python -m app.app
```

Once the server is running, open your browser and go to:

```
http://127.0.0.1:5000
```

to access the training interface.

---

## 4. API Endpoints

| Method | Path       | Description                                  |
|--------|------------|----------------------------------------------|
| POST   | `/train`   | Trigger model training; returns training metrics as JSON. |
| POST   | `/predict` | Submit JSON payload of feature values to receive a prediction. |

### Example: `/predict` Payload
```json
{
  "set_vals": [2, 2, 2, 2, 3, 3, 1, 1, 1, 2, 2]
}
```

Response:
```json
{
   "predicted_ust": 0,
   "probability_ust": 0.3333
}
```# model-test
