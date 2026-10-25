# 🤟 Running Signora on your computer

Signora is a Flask (Python) app with a TensorFlow + MediaPipe model.
Your **browser's webcam** sends frames to the local server, which recognises the ISL sign.

---

## 1. Requirements

| | |
|---|---|
| **Python** | **3.11 exactly** (TensorFlow 2.13 doesn't support 3.12 / 3.13) |
| **OS** | Windows 10/11 (64-bit), macOS 12+ (Intel or Apple Silicon), Linux |
| **RAM / disk** | ~2 GB free RAM, ~2.5 GB disk for dependencies |
| **Browser** | Chrome / Edge recommended for camera access and speech output |

**Get Python 3.11:**
- **Windows:** <https://www.python.org/downloads/release/python-3119/> → *Windows installer (64-bit)* → tick **“Add python.exe to PATH”**
- **macOS:** `brew install python@3.11`
- **Ubuntu/Debian:** `sudo apt install python3.11 python3.11-venv`

---

## 2. Quick start (one command)

Unzip `signora.zip`, open a terminal **inside the `isl-mirror` folder**, then:

**Windows** – double-click `run_local.bat` (or run it in Command Prompt)

**macOS / Linux**
```bash
bash run_local.sh
```

The first run creates a `venv` folder and installs everything (5–10 min).
After that it starts in a few seconds and opens **http://localhost:5000** automatically.
Allow camera access when the browser asks.

Stop the server with **Ctrl + C**.

---

## 3. Manual setup (if you prefer)

```bash
# 1. create & activate a virtual environment
python3.11 -m venv venv            # Windows: py -3.11 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

# 2. install dependencies
pip install --upgrade pip
pip install -r requirements-local.txt

# 3. run
python run_local.py
```

Open **http://localhost:5000**.

Options (environment variables):
- `PORT=8000 python run_local.py` – use another port
- `NO_BROWSER=1 python run_local.py` – don't auto-open the browser

---

## 4. Run with Docker (alternative)

Same setup as the Hugging Face Space:

```bash
docker build -t signora .
docker run -p 7860:7860 signora
```
Open **http://localhost:7860**.

---

## 5. Pages

| URL | What it does |
|---|---|
| `/` | Sign to Text (live recognition with speech output) |
| `/text-to-sign` | Text to Sign (animated sign sequence) |

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `No matching distribution found for tensorflow==2.13.0` | You're not on Python 3.11. Delete the `venv` folder and recreate it with 3.11. |
| Camera doesn't start | Use `http://localhost:5000` (not your IP address) – browsers only allow the camera on localhost or HTTPS. Close other apps using the webcam. |
| Port 5000 already in use (common on macOS – AirPlay Receiver) | `PORT=5050 bash run_local.sh` or turn off *AirPlay Receiver* in System Settings. |
| Speak Text is silent on Linux | `sudo apt install espeak-ng libespeak1` (Windows/macOS work out of the box). |
| Translate feature errors | Needs an internet connection (uses Google Translate). |
| Windows: `py` not found | Reinstall Python 3.11 and tick “Add python.exe to PATH”. |

To reinstall from scratch, delete the `venv` folder and run the script again.
