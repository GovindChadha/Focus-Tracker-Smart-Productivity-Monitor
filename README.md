# 📱 Focus Tracker — “Phone vs Work” Detector

A tiny OpenCV app that uses your webcam to check if you're **working** or **on your phone**.  
If a phone (hand/skin region) covers your face, it flags you as **Using Phone**; otherwise **Working**.

---

## 🔧 How it works (in plain English)
- Uses a **frontal face Haar cascade** to find your face in each frame.  
- Uses simple **skin-color segmentation** + contour size to find a hand/phone-like region.  
- If a face **and** a large skin/hand region are present, we treat you as **Using Phone** (covering face).  
- Otherwise you’re **Working**.

(See the code in `motion_activity_detector.py` for the exact logic and thresholds.)

---

## 📁 Project Files
- `motion_activity_detector.py` — main script (OpenCV loop, face + hand detection).  
- `haarcascade_frontalface_default.xml` — cascade for face detection.  
- `hand.xml` — cascade file for hands (optional backup; app primarily uses skin segmentation).

---

## 🖥️ Requirements
- Python **3.9+**
- A webcam
- Packages (see `requirements.txt`)

