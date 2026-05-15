# Ensemble Lane Runtime

Separate deployment candidate that combines:

```text
lane_mobilenetv3_large.tflite
lane_nvidia.tflite
```

Copy these files to the Pi model folder when testing this candidate:

```text
model.py
lane_mobilenetv3_large.tflite
lane_nvidia.tflite
arrow_model.tflite
obstacle_detector.tflite
```

The model loads both TFLite files and blends their angle predictions.

Main tuning constants are near the top of `model.py`:

```python
ENSEMBLE_MODE = "weighted"
MOBILENET_WEIGHT = 0.65
NVIDIA_WEIGHT = 0.35
```

Alternative modes:

```python
ENSEMBLE_MODE = "agreement"
ENSEMBLE_MODE = "conditional"
```

The debug stream displays:

```text
ens  = blended angle
mob  = MobileNet angle
nvid = NVIDIA angle
```

For a clean comparison, keep the runtime right-turn boost disabled:

```python
RIGHT_TURN_BOOST = 0.0
RIGHT_TURN_MIN_ANGLE = 0.0
```

Arrow and obstacle detection are included in observer mode by default:

```python
ENABLE_ARROW_CONTROL = False
ENABLE_OBSTACLE_STOP = False
```

The debug stream shows arrow confidence and obstacle boxes. Once detections look reliable, enable control separately:

```python
ENABLE_OBSTACLE_STOP = True
ENABLE_ARROW_CONTROL = True
```
