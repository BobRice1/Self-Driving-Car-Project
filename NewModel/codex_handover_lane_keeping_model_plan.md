# Codex Handover Plan: Retrain and Rebuild `model.py` for Raspberry Pi Car Left-Lane Keeping

## 1. Objective

Build a new end-to-end lane-keeping system for the Raspberry Pi car that can reliably drive around the oval track and similar tracks while staying inside the **left-hand lane at all times**.

The current system appears to fail because OpenCV geometry is acting as the primary steering source. On bends, OpenCV can become confidently wrong and steer the car off the track. The new system should be redesigned so that the trained model is the primary driver, while OpenCV is only used as a light safety monitor or debugging aid.

The steering convention is:

```text
50  = hard left
90  = straight
120 = hard right
```

The current data consists of approximately 15,000 labelled images from the Pi car, with each frame labelled by:

```text
image → steering angle + speed
```

We are allowed to retrain every aspect of the model.

---

## 2. Core Design Decision

Use a **machine-learning-primary architecture**, not OpenCV-primary control.

The final runtime system should be:

```text
Camera frame
   ↓
Preprocessing matched to training
   ↓
TFLite model predicts steering angle, optionally speed
   ↓
Prediction sanity checks
   ↓
Temporal smoothing
   ↓
Steering rate limiting
   ↓
Optional OpenCV safety monitor
   ↓
Rule-based speed control
   ↓
Return angle, speed
```

The OpenCV layer should not calculate the normal driving angle. It should only apply a small correction or reduce speed when the car is clearly near the dashed centre line or outer solid line.

---

## 3. Problems With the Current Approach

The current `model.py` should be treated as a useful prototype, not the final design.

Likely issues:

1. OpenCV is the primary controller.
2. The TFLite lane model is only used as fallback.
3. OpenCV can return `lane.ok = True` while detecting the wrong geometry.
4. Fixed x-threshold assumptions break on oval bends.
5. Speed remains too high if OpenCV is wrong but still confident.
6. Steering can become unstable if the perceived target jumps.
7. The model output scaling may be ambiguous.

The main design correction is:

```text
Current:
OpenCV decides steering → model only used if OpenCV fails

New:
Model decides steering → OpenCV only performs safety checks
```

---

## 4. Model Training Strategy

### 4.1 Train the model to output real steering angle

The preferred model output should be direct steering angle in the car's actual range:

```text
50–120
```

This avoids confusing runtime conversions.

Preferred labels:

```text
image → steering_angle
```

Optional second output:

```text
image → speed
```

However, speed should initially be controlled with a rule-based policy during deployment. The model speed output can be logged or used as a weak suggestion after steering is stable.

---

### 4.2 Recommended model output heads

Use one of these options:

#### Option A: Steering-only model

```text
Input: image
Output: steering angle
```

This is simplest and preferred for the first stable version.

#### Option B: Multi-task model

```text
Input: image
Outputs:
- steering angle
- speed
```

Use loss weighting:

```text
steering loss weight = 1.0
speed loss weight    = 0.2
```

Reason: steering accuracy is much more important than speed imitation.

#### Option C: Steering + uncertainty

```text
Input: image
Outputs:
- steering angle
- confidence or uncertainty
```

This is useful later, but not required for the first working version.

---

## 5. Dataset Preparation

The current images are not separated by track folders. Do not randomly split individual images, because adjacent frames from the same lap are near-duplicates and will leak between train and validation.

### 5.1 Sort the dataset

Sort by one of:

```text
filename order
timestamp
CSV row order
creation time
```

The goal is to preserve driving sequence order.

---

### 5.2 Create chunk-based splits

Split the dataset into continuous chunks, for example 500 frames per chunk.

Example:

```python
import numpy as np
import pandas as pd

df = pd.read_csv("driving_log.csv")
df = df.sort_values("image_path").reset_index(drop=True)

chunk_size = 500
df["chunk_id"] = np.arange(len(df)) // chunk_size

chunks = df["chunk_id"].unique()

test_chunks = chunks[::6]
val_chunks = chunks[3::6]

test = df[df["chunk_id"].isin(test_chunks)]
val = df[df["chunk_id"].isin(val_chunks)]
train = df[~df["chunk_id"].isin(np.concatenate([test_chunks, val_chunks]))]

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)
```

This is not as good as true per-track splitting, but it is much better than random frame splitting.

---

### 5.3 Create a bend-specific validation set

The car currently fails halfway around the oval bend. Therefore, create a small manually selected validation subset containing:

```text
bend entry
mid-bend
bend exit
frames close to dashed centre line
frames close to outer boundary
recovery situations
```

This can be done by manually tagging 200–500 frames.

Create:

```text
val_bend.csv
```

This should be reported separately during evaluation.

---

### 5.4 Optional pseudo-track labelling

If we want track-aware splits later, infer pseudo-track groups:

1. Resize all images to 32×32 or 64×64.
2. Flatten or use embeddings from a small CNN.
3. Run PCA or UMAP.
4. Cluster with KMeans into 3 groups.
5. Manually inspect samples from each cluster.
6. Add a `pseudo_track_id` column.

Do not blindly trust clustering. It may separate by lighting or camera angle rather than track.

---

## 6. Data Cleaning

Before retraining, inspect and clean labels.

### 6.1 Remove broken frames

Remove images that are:

```text
missing
corrupt
blank
severely blurred
wrong resolution
not from the car camera
```

### 6.2 Check steering distribution

Plot the label distribution.

```python
import matplotlib.pyplot as plt

plt.hist(df["steering"], bins=50)
plt.xlabel("Steering angle")
plt.ylabel("Count")
plt.show()
```

Expect many frames near 90. If the model sees too many straight frames, it may learn to understeer on bends.

---

### 6.3 Rebalance turning examples

Use one or more of:

```text
downsample near-straight frames
oversample turning frames
sample weighting
bend-specific augmentation
```

Suggested sample weight:

```python
def steering_weight(angle):
    return 1.0 + 2.0 * abs(angle - 90) / 40.0
```

This makes turning frames more important.

---

### 6.4 Add recovery data if possible

If the car fails when slightly off-centre, the dataset may not contain enough recovery examples.

Collect extra data where the car is deliberately positioned:

```text
slightly too close to dashed centre line
slightly too close to outer boundary
entering the bend too wide
entering the bend too tight
mid-bend corrections
bend exit corrections
```

The labels should show how to recover:

```text
too close to dashed line → steer left  → angle < 90
too close to outer line  → steer right → angle > 90
```

This is crucial. A behaviour cloning model trained only on perfect driving often cannot recover from small mistakes.

---

## 7. Data Augmentation

Use realistic augmentation only.

Recommended augmentations:

```text
brightness changes
contrast changes
small shadows
slight blur
small rotations
small horizontal/vertical shifts
slight perspective changes
```

Be careful with horizontal flipping. Since this is left-lane keeping, flipping may create invalid training examples unless the steering label and lane semantics are corrected carefully. For the first model, avoid horizontal flipping.

### 7.1 Steering-aware horizontal shift

Horizontal shifting is useful because it teaches recovery.

If the image is shifted left or right, adjust the steering label.

Example:

```python
# If image is shifted right, the car appears too far left,
# so it should steer more right: increase angle.
# If image is shifted left, the car appears too far right,
# so it should steer more left: decrease angle.

adjusted_angle = angle + shift_pixels * correction_gain
```

Tune the sign and gain using visual inspection.

---

## 8. Model Architecture

Use a small CNN suitable for Raspberry Pi inference.

### 8.1 First recommended architecture

Use a lightweight NVIDIA-style CNN:

```text
Input image
↓
Conv2D 24 filters, 5×5, stride 2, ReLU
↓
Conv2D 36 filters, 5×5, stride 2, ReLU
↓
Conv2D 48 filters, 5×5, stride 2, ReLU
↓
Conv2D 64 filters, 3×3, ReLU
↓
Conv2D 64 filters, 3×3, ReLU
↓
Flatten
↓
Dense 100, ReLU
↓
Dense 50, ReLU
↓
Dense 10, ReLU
↓
Dense 1 steering angle
```

Input size:

```text
160×80 RGB
```

or:

```text
200×66 RGB
```

The exact input size must match runtime preprocessing.

---

### 8.2 Alternative: MobileNetV2 small head

If inference speed is acceptable:

```text
MobileNetV2 width multiplier 0.35 or 0.5
global average pooling
dense head
steering output
```

This may generalise better, but the NVIDIA-style model is simpler and faster.

---

## 9. Training Loss

Use a robust steering loss.

Recommended:

```text
Huber loss
```

instead of plain MSE, because it is less sensitive to noisy labels.

Example:

```python
loss = tf.keras.losses.Huber(delta=5.0)
```

Use sample weights based on turning intensity:

```python
weight = 1.0 + 2.0 * abs(angle - 90) / 40.0
```

If using speed as a second output:

```python
loss = {
    "steering": Huber(delta=5.0),
    "speed": Huber(delta=3.0),
}

loss_weights = {
    "steering": 1.0,
    "speed": 0.2,
}
```

---

## 10. Training Metrics

Do not only track overall validation loss. Track:

```text
steering MAE overall
steering MAE on bend-specific validation set
MAE for left steering examples
MAE for right steering examples
MAE for near-straight examples
percentage of predictions clipped at runtime limits
prediction smoothness over frame sequences
```

Useful bins:

```text
hard left:      50–75
moderate left: 75–85
straight:      85–95
moderate right:95–105
hard right:    105–120
```

The bend failure will not be visible if we only look at average MAE.

---

## 11. Offline Sequence Evaluation

Before deploying, replay validation sequences frame by frame.

For each sequence:

```text
plot labelled steering vs predicted steering
plot smoothed steering vs raw prediction
plot speed policy
identify sudden prediction jumps
```

Look specifically at the oval bend.

Acceptance criteria before driving:

```text
no major steering sign errors
no sudden jumps from left to right on bend
prediction moves smoothly through bend
predicted steering magnitude is sufficient for corners
```

---

## 12. Export to TFLite

After training:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("lane_model.tflite", "wb") as f:
    f.write(tflite_model)
```

Use float32 first if possible. Quantise later only after the behaviour works.

Deployment order:

```text
1. float32 TFLite model
2. dynamic range quantisation
3. full int8 quantisation only if needed
```

Do not debug training, control, and quantisation problems all at once.

---

## 13. New `model.py` Runtime Design

The runtime file should expose:

```python
class Model:
    def predict(self, image: np.ndarray) -> tuple[int, int]:
        ...
```

### 13.1 Constants

Start conservative:

```python
ANGLE_MIN = 50
ANGLE_MAX = 120
ANGLE_STRAIGHT = 90

ANGLE_RUNTIME_MIN = 65
ANGLE_RUNTIME_MAX = 115

BASE_SPEED = 16
SLOW_SPEED = 13
VERY_SLOW_SPEED = 10

CROP_TOP_RATIO = 0.35

MODEL_OUTPUT_MODE = "angle"  # preferred

STEERING_EMA_ALPHA = 0.35
MAX_STEERING_DELTA = 7

USE_OPENCV_SAFETY = False  # start disabled for testing model-only
SAFETY_CORRECTION = 6
```

Once model-only behaviour is stable, enable safety.

---

### 13.2 Runtime control logic

Pseudo-code:

```python
raw_angle, raw_speed = model.predict(image)

angle = convert_to_angle(raw_angle)
angle = sanity_check(angle)

if USE_OPENCV_SAFETY:
    safety = safety_monitor.check(image)
    angle += safety.correction
else:
    safety = inactive

angle = rate_limit(angle, last_angle)
angle = ema_smooth(angle, last_angle)
angle = clip(angle, runtime_min, runtime_max)

speed = choose_speed(angle, raw_speed, safety)

last_angle = angle
return angle, speed
```

---

## 14. OpenCV Safety Monitor

OpenCV should only be used as a weak safety layer.

It should check:

```text
Is the car too close to the dashed centre line?
Is the car too close to the outer solid line?
Is the lower field of view missing track markings?
```

It should return:

```python
@dataclass
class SafetyStatus:
    active: bool
    correction: float
    reason: str
    outer_x: Optional[float]
    dashed_x: Optional[float]
    confidence: float
```

Correction convention:

```text
too close to dashed centre line → steer left  → decrease angle
too close to outer solid line   → steer right → increase angle
```

Implementation:

```python
if too_close_to_dashed:
    correction -= SAFETY_CORRECTION

if too_close_to_outer:
    correction += SAFETY_CORRECTION
```

Do not allow OpenCV to fully override the model unless an emergency stop is needed.

---

## 15. Speed Policy

At first, use rule-based speed.

```python
steering_demand = abs(angle - 90)

if steering_demand > 22:
    speed = VERY_SLOW_SPEED
elif steering_demand > 14:
    speed = SLOW_SPEED
else:
    speed = BASE_SPEED

if safety.active:
    speed = min(speed, SLOW_SPEED)
```

Do not increase speed until the car completes multiple clean laps.

Suggested tuning sequence:

```text
BASE_SPEED 12 → 14 → 16 → 18 → 20
```

---

## 16. Testing Protocol

### Stage 1: Offline model test

Run saved frames through the model.

Log:

```text
raw model output
converted angle
smoothed angle
label angle
absolute error
```

Check that:

```text
straight frames predict around 90
left turns predict below 90
right turns predict above 90
no output is NaN
no output is wildly outside 50–120
```

---

### Stage 2: Model-only live test

Set:

```python
USE_OPENCV_SAFETY = False
BASE_SPEED = 12
SLOW_SPEED = 10
VERY_SLOW_SPEED = 8
ANGLE_RUNTIME_MIN = 70
ANGLE_RUNTIME_MAX = 110
```

Goal:

```text
Does the ML model alone get further around the bend than the current OpenCV-primary setup?
```

If yes, OpenCV was the main failure source.

---

### Stage 3: Widen steering range

If model-only is stable but cannot turn enough:

```python
ANGLE_RUNTIME_MIN = 65
ANGLE_RUNTIME_MAX = 115
```

Then later:

```python
ANGLE_RUNTIME_MIN = 60
ANGLE_RUNTIME_MAX = 118
```

Do not widen limits if the car is twitchy.

---

### Stage 4: Enable weak safety

Set:

```python
USE_OPENCV_SAFETY = True
SAFETY_CORRECTION = 4
```

If safety helps, increase to:

```python
SAFETY_CORRECTION = 6
```

If safety causes sudden steering jumps, disable it or make it only slow the car instead of steering.

---

### Stage 5: Increase speed

Only after the car completes multiple laps:

```text
BASE_SPEED 12 → 14 → 16 → 18 → 20
```

Keep bend speed lower than straight speed.

---

## 17. Debug Logging

Print every 10 frames:

```text
raw_angle
converted_angle
final_angle
speed
safety_reason
safety_correction
outer_x
dashed_x
```

Example:

```python
print(
    f"[model] raw={raw_angle:.2f} converted={model_angle:.1f} "
    f"final={angle} speed={speed} "
    f"safety={safety.reason} corr={safety.correction:.1f} "
    f"outer={safety.outer_x} dashed={safety.dashed_x}"
)
```

---

## 18. Runtime `model.py` Acceptance Criteria

The new file is acceptable when:

```text
1. It loads `lane_model.tflite`.
2. It supports float and quantised TFLite inputs/outputs.
3. It treats the model prediction as the primary steering command.
4. It clips steering to safe runtime limits.
5. It rate-limits steering changes.
6. It smooths steering over time.
7. It uses rule-based speed control.
8. It can run with OpenCV safety disabled.
9. OpenCV safety, when enabled, only applies small corrections.
10. It logs enough information to diagnose failures.
```

---

## 19. Model Acceptance Criteria

The retrained model is acceptable when:

```text
1. Steering output is in the correct 50–120 scale.
2. Straight frames predict close to 90.
3. Left-turn frames predict below 90.
4. Right-turn frames predict above 90.
5. Bend-specific validation error is acceptable.
6. Predictions are smooth over frame sequences.
7. The car can complete the oval bend at low speed.
8. The car can complete multiple full laps without leaving the left lane.
```

---

## 20. Suggested Implementation Order for Codex

### Step 1

Build data loading and chunk-based split scripts.

Outputs:

```text
train.csv
val.csv
test.csv
val_bend.csv if available
```

### Step 2

Build training script.

Outputs:

```text
trained Keras model
training curves
offline validation plots
```

### Step 3

Export float32 TFLite model.

Output:

```text
lane_model.tflite
```

### Step 4

Build new ML-primary `model.py`.

Start with:

```python
USE_OPENCV_SAFETY = False
BASE_SPEED = 12
ANGLE_RUNTIME_MIN = 70
ANGLE_RUNTIME_MAX = 110
```

### Step 5

Live test model-only.

### Step 6

Tune steering range and smoothing.

### Step 7

Enable weak OpenCV safety if needed.

### Step 8

Increase speed gradually.

---

## 21. Important Notes

Do not optimise for lap speed first. Optimise for:

```text
staying inside the left-hand lane
stable steering
recovering from small errors
reliable bend handling
```

The car failing halfway around the bend is more important than average validation loss. All training and testing should focus on whether the model can handle the bend sequence smoothly and recover if slightly off-centre.

The final runtime should never allow one bad frame to produce a huge steering command. Rate limiting and smoothing are mandatory.

