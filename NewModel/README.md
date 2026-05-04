# NewModel lane keeping build

This folder implements the handover plan in `codex_handover_lane_keeping_model_plan.md`.

The training target is the real Pi car steering angle:

- `50` = hard left
- `90` = straight
- `120` = hard right

## 1. Prepare splits

```powershell
python NewModel\prepare_data.py `
  --csv-data data `
  --drive-frames car\data\drive_frames `
  --out-dir NewModel\splits
```

This combines:

- `data/train.csv` with `data/training_data`
- filename-labelled frames in `car/data/drive_frames`, using names like `timestamp_angle_speed.png`

The script writes chunk-based `train.csv`, `val.csv`, `test.csv`, and `dataset_summary.csv`.

## 2. Train

Use the main project `.venv` for training. It has CUDA PyTorch installed; on this machine `--device auto` detects the RTX 3060 Laptop GPU.

Current deployed architecture is MobileNetV3-Large:

```powershell
.\.venv\Scripts\python.exe NewModel\train_lane_model.py `
  --arch mobilenet_v3_large `
  --splits-dir NewModel\splits `
  --out-dir NewModel\runs\lane_mobilenetv3_large_right_weighted `
  --epochs 25 `
  --batch-size 128 `
  --num-workers 0 `
  --device auto `
  --selection-metric bend_mae `
  --right-turn-weight 3.0 `
  --drive-frame-weight 1.5
```

The best checkpoint is saved as:

```text
NewModel/runs/lane_mobilenetv3_large_right_weighted/best.pt
```

The model predicts direct car steering angle, not normalized steering.

Older NVIDIA CNN checkpoints are still available for fallback/comparison:

```text
NewModel/runs/lane_nvidia_right_weighted_balanced/best.pt
NewModel/runs/lane_nvidia_bend_selected/best.pt
NewModel/runs/lane_nvidia/best.pt
```

## 3. Export

Export ONNX:

```powershell
.\.venv\Scripts\python.exe NewModel\export_lane_model.py `
  --checkpoint NewModel\runs\lane_mobilenetv3_large_right_weighted\best.pt `
  --onnx NewModel\runs\lane_mobilenetv3_large_right_weighted\lane_model.onnx
```

Convert ONNX to TFLite with `.venv_tflite`:

```powershell
.\.venv_tflite\Scripts\python.exe -m car.training.convert_onnx_to_tflite `
  --onnx NewModel\runs\lane_mobilenetv3_large_right_weighted\lane_model.onnx `
  --output NewModel\lane_model.tflite
```

Direct `--tflite` export is also supported by `export_lane_model.py` if a working `litert-torch` stack is installed, but this repo's `.venv_tflite` conversion route is the verified local path.

## 4. Deploy

Copy these files to the Pi model folder:

```text
NewModel/model.py
NewModel/lane_model.tflite
```

The runtime is ML-primary. OpenCV safety is disabled by default and only applies small corrections when enabled.

## 5. Pi testing and debugging knobs

The Pi runtime reads environment variables at startup. Change them before starting the car program.

### Live visual debug stream

Enable the MJPEG debug stream:

```bash
export PICAR_DEBUG_STREAM=1
export PICAR_DEBUG_PORT=8080
```

Then open this from a phone/laptop on the same network:

```text
http://<pi-ip>:8080
```

The stream shows the live frame, model angle, final smoothed angle, speed, OpenCV safety status, and line mask. Keep it off for clean latency tests:

```bash
export PICAR_DEBUG_STREAM=0
```

If the Pi struggles, reduce stream load:

```bash
export PICAR_DEBUG_FPS=3
export PICAR_DEBUG_JPEG_QUALITY=55
```

### Steering range

Use this if the model points the correct way but the car physically does not turn enough:

```bash
export PICAR_ANGLE_RUNTIME_MIN=65
export PICAR_ANGLE_RUNTIME_MAX=115
```

More conservative:

```bash
export PICAR_ANGLE_RUNTIME_MIN=75
export PICAR_ANGLE_RUNTIME_MAX=105
```

Defaults are `70` and `110`.

### Steering smoothing

If steering is twitchy:

```bash
export PICAR_STEERING_EMA_ALPHA=0.25
export PICAR_MAX_STEERING_DELTA=5
```

If the car reacts too slowly into bends:

```bash
export PICAR_STEERING_EMA_ALPHA=0.45
export PICAR_MAX_STEERING_DELTA=9
```

`PICAR_STEERING_EMA_ALPHA` controls how much of the latest prediction is used. Higher is more responsive. `PICAR_MAX_STEERING_DELTA` limits how many angle units can change per frame.

### Speed

For first tests, keep speed low:

```bash
export PICAR_BASE_SPEED=10
export PICAR_SLOW_SPEED=8
export PICAR_VERY_SLOW_SPEED=6
```

If it is stable:

```bash
export PICAR_BASE_SPEED=14
export PICAR_SLOW_SPEED=11
export PICAR_VERY_SLOW_SPEED=8
```

The runtime automatically slows down when the requested steering angle is far from straight.

### OpenCV safety correction

OpenCV safety is separate from the model. It looks for black track markings on the white paper and applies a small correction when the car drifts near a boundary.

Enable it:

```bash
export PICAR_USE_OPENCV_SAFETY=1
export PICAR_SAFETY_CORRECTION=4
```

If it helps but feels weak:

```bash
export PICAR_SAFETY_CORRECTION=6
```

If it fights the model or causes oscillation:

```bash
export PICAR_USE_OPENCV_SAFETY=0
```

### Logging

Print less often:

```bash
export PICAR_DEBUG_EVERY=30
```

Print more often:

```bash
export PICAR_DEBUG_EVERY=5
```

## 6. Experimental arrow and obstacle runtime

`NewModel/experimental_event_model` is a separate deployment bundle for figure-of-8/T-junction experiments. Copy the whole folder contents to the Pi model folder:

```text
model.py
lane_model.tflite
arrow_model.tflite
obstacle_detector.tflite
```

Start with observer mode so lane keeping still controls the car and the event models only report detections:

```bash
export PICAR_DEBUG_STREAM=1
export PICAR_ENABLE_ARROW_CONTROL=0
export PICAR_ENABLE_OBSTACLE_STOP=0
```

If obstacle detections look correct in the stream, enable stopping:

```bash
export PICAR_ENABLE_OBSTACLE_STOP=1
export PICAR_OBSTACLE_CONFIDENCE=0.4
```

If arrow detections look correct, enable arrow turns:

```bash
export PICAR_ENABLE_ARROW_CONTROL=1
export PICAR_ARROW_CONFIDENCE=0.92
export PICAR_ARROW_CONFIRM_FRAMES=2
export PICAR_ARROW_TURN_FRAMES=20
```

Useful arrow turn tuning:

```bash
export PICAR_ARROW_LEFT_ANGLE=70
export PICAR_ARROW_RIGHT_ANGLE=110
```

Event models run every 5 frames by default. Check more often at the cost of extra inference load:

```bash
export PICAR_EVENT_INTERVAL=3
```

## 7. Quick test sequence

1. Run lane-only with debug stream on and low speed.
2. If it understeers, widen steering range before changing the model.
3. If it is twitchy, lower `PICAR_STEERING_EMA_ALPHA` or `PICAR_MAX_STEERING_DELTA`.
4. If it misses bends because it reacts late, raise `PICAR_STEERING_EMA_ALPHA` slightly.
5. Enable OpenCV safety only after the base model is reasonably stable.
6. Test `experimental_event_model` in observer mode before enabling arrow/obstacle control.
