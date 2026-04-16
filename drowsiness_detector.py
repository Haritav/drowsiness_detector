"""
Driver Drowsiness Detection System
===================================
Real-Time CNN-Based Eye State Classification

Authors : Avani Jain, Harita Venkatesan, Jatin Rajabhoj, Jayaharsh Kosanam
Institute: RV University, Bangalore, India

Usage:
    python drowsiness_detector.py

First run trains and saves the model.
Subsequent runs load the saved model and open webcam directly.
Press Q to quit.
"""

import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import time

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
IMG_SIZE        = (24, 24)
BATCH_SIZE      = 64
EPOCHS          = 10
ALERT_THRESHOLD = 20
MODEL_PATH      = "drowsiness_model.h5"
DATASET_PATH    = "dataset"

# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────
def train_model():
    print("\n" + "="*55)
    print("  TRAINING LIGHTWEIGHT CNN")
    print("="*55)

    gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_data = gen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        subset='training'
    )
    val_data = gen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        subset='validation'
    )

    print(f"  Classes : {train_data.class_indices}")
    print(f"  Train   : {train_data.samples}")
    print(f"  Val     : {val_data.samples}\n")

    model = Sequential([
        Conv2D(16, (3,3), activation='relu',
               input_shape=(24, 24, 1)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        steps_per_epoch=100,
        validation_data=val_data,
        validation_steps=30,
        callbacks=[EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )]
    )

    acc = max(history.history['val_accuracy']) * 100
    print(f"\n  Done! Best accuracy: {acc:.2f}%")
    model.save(MODEL_PATH)
    print(f"  Saved to {MODEL_PATH}\n")
    return model


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
def draw_ui(frame, status, confidence, closed_counter, fps):
    h, w = frame.shape[:2]

    color = {
        "ALERT":   (34,  139, 34),
        "WARNING": (0,   140, 255),
        "DROWSY":  (0,   0,   220),
        "NO FACE": (80,  80,  80),
    }.get(status, (80, 80, 80))

    # top bar
    cv2.rectangle(frame, (0, 0), (w, 75), color, -1)
    text = {
        "ALERT":   "ALERT  —  Eyes Open  |  Stay Safe",
        "WARNING": "WARNING  —  Eyes Closing  |  Stay Focused",
        "DROWSY":  "DROWSY ALERT!  —  Please Pull Over!",
        "NO FACE": "No Face Detected  —  Adjust Camera",
    }.get(status, "")
    cv2.putText(frame, text, (15, 52),
        cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2)

    # red flash
    if status == "DROWSY":
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,h), (0,0,180), -1)
        frame = cv2.addWeighted(ov, 0.2, frame, 0.8, 0)

    # bottom panel
    cv2.rectangle(frame, (0, h-90), (w, h), (15,15,15), -1)
    cv2.line(frame, (0, h-90), (w, h-90), color, 2)

    # confidence bar
    cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
        (15, h-62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    bx = w//2 - 20
    cv2.rectangle(frame, (15, h-48), (bx, h-28), (50,50,50), -1)
    cv2.rectangle(frame, (15, h-48),
        (15 + int((bx-15)*confidence), h-28), color, -1)

    # counter bar
    cv2.putText(frame,
        f"Closed Frames: {closed_counter}/{ALERT_THRESHOLD}",
        (w//2+10, h-62),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cx = w - 20
    cv2.rectangle(frame, (w//2+10, h-48), (cx, h-28), (50,50,50), -1)
    ratio = min(closed_counter / ALERT_THRESHOLD, 1.0)
    fc = (34,139,34) if ratio < 0.5 \
        else (0,140,255) if ratio < 1.0 \
        else (0,0,220)
    cv2.rectangle(frame,
        (w//2+10, h-48),
        (w//2+10 + int((cx - w//2 - 10)*ratio), h-28), fc, -1)

    # fps + footer
    cv2.putText(frame, f"FPS: {fps:.0f}",
        (w-110, h-62),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130,130,130), 1)
    cv2.putText(frame,
        "Driver Drowsiness Detection  |  CNN  |  Q to quit",
        (15, h-8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70,70,70), 1)

    return frame


# ─────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────
def run_detection(model):
    print("\n" + "="*55)
    print("  LIVE DETECTION STARTED — Press Q to quit")
    print("="*55 + "\n")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        'haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        'haarcascade_eye.xml'
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  854)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("  ERROR: webcam not found.")
        return

    closed_counter = 0
    status         = "NO FACE"
    confidence     = 0.0
    frame_count    = 0
    prev_time      = time.time()

    # warm up model
    dummy = np.zeros((1, 24, 24, 1), dtype="float32")
    model(dummy, training=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # run detection every 3rd frame only
        if frame_count % 3 == 0:

            small      = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            faces_small = face_cascade.detectMultiScale(
                gray_small, 1.1, 4, minSize=(60, 60))
            faces = [(int(x/0.4), int(y/0.4),
                      int(w/0.4), int(h/0.4))
                     for (x,y,w,h) in faces_small]

            if len(faces) == 0:
                status         = "NO FACE"
                confidence     = 0.0
                closed_counter = 0

            else:
                fx, fy, fw, fh = faces[0]
                fx = max(0, fx)
                fy = max(0, fy)
                fw = min(fw, frame.shape[1] - fx)
                fh = min(fh, frame.shape[0] - fy)

                cv2.rectangle(frame,
                    (fx, fy), (fx+fw, fy+fh), (100,200,255), 2)
                cv2.putText(frame, "Face",
                    (fx, fy-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (100,200,255), 1)

                face_roi         = frame[fy:fy+fh, fx:fx+fw]
                eye_h            = int(fh * 0.55)
                eye_color_region = face_roi[0:eye_h, :]
                eye_gray_region  = cv2.cvtColor(
                    eye_color_region, cv2.COLOR_BGR2GRAY)

                eyes = eye_cascade.detectMultiScale(
                    eye_gray_region,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(20, 20)
                )

                if len(eyes) == 0:
                    closed_counter += 1
                    confidence      = 0.9

                else:
                    preds = []
                    for (ex, ey, ew, eh) in eyes[:2]:
                        crop = eye_color_region[ey:ey+eh, ex:ex+ew]
                        if crop.size == 0:
                            continue

                        crop = cv2.resize(crop, (24, 24))
                        crop = cv2.cvtColor(
                            crop, cv2.COLOR_BGR2GRAY)
                        crop = crop.astype("float32") / 255.0
                        crop = crop.reshape(1, 24, 24, 1)

                        p = model(crop, training=False).numpy()[0][0]
                        preds.append(p)

                        ec = (0,220,0) if p > 0.5 else (0,0,220)
                        lb = f"Open {p*100:.0f}%" \
                            if p > 0.5 \
                            else f"Closed {(1-p)*100:.0f}%"
                        cv2.rectangle(eye_color_region,
                            (ex,ey),(ex+ew,ey+eh), ec, 2)
                        cv2.putText(eye_color_region, lb,
                            (ex, ey-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, ec, 1)

                    if not preds:
                        closed_counter += 1
                    else:
                        avg = np.mean(preds)
                        confidence = avg if avg > 0.5 else 1 - avg
                        if avg < 0.5:
                            closed_counter += 1
                        else:
                            closed_counter = max(0, closed_counter - 2)

                status = "ALERT"   if closed_counter == 0 \
                    else "WARNING" if closed_counter < ALERT_THRESHOLD \
                    else "DROWSY"

        now       = time.time()
        fps       = 1 / (now - prev_time + 1e-6)
        prev_time = now

        frame = draw_ui(
            frame, status, confidence, closed_counter, fps)
        cv2.imshow("Driver Drowsiness Detection", frame)

        print(f"\r  {status:<10} | "
              f"Conf: {confidence*100:4.0f}% | "
              f"Closed: {closed_counter:2d}/{ALERT_THRESHOLD} | "
              f"FPS: {fps:4.0f}",
              end="", flush=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n\n  Stopped. Goodbye!")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  DRIVER DROWSINESS DETECTION")
    print("  Lightweight CNN  |  OpenCV  |  Real-Time")
    print("="*55)

    if os.path.exists(MODEL_PATH):
        print(f"\n  Loading saved model: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("  Loaded!")
    else:
        print("\n  No model found — training now...")
        model = train_model()

    run_detection(model)
