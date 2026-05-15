# Experimental Lane + Events Runtime

This folder is a separate Pi deployment bundle for testing arrow and obstacle models without changing the current lane-only `car/models/lane_keeping/current/model.py`.

## Files to copy to the Pi

```text
model.py
lane_model.tflite
arrow_model.tflite
obstacle_detector.tflite
```

## Safe observer test

This runs lane keeping normally and only reports arrow/obstacle detections in logs and the debug stream.

```bash
export PICAR_DEBUG_STREAM=1
export PICAR_DEBUG_PORT=8080
export PICAR_ENABLE_ARROW_CONTROL=0
export PICAR_ENABLE_OBSTACLE_STOP=0
```

Open:

```text
http://<pi-ip>:8080
```

## Enable obstacle stop

```bash
export PICAR_ENABLE_OBSTACLE_STOP=1
export PICAR_OBSTACLE_CONFIDENCE=0.4
```

## Enable arrow turns

```bash
export PICAR_ENABLE_ARROW_CONTROL=1
export PICAR_ARROW_CONFIDENCE=0.92
export PICAR_ARROW_CONFIRM_FRAMES=2
export PICAR_ARROW_TURN_FRAMES=20
```

Useful tuning:

```bash
export PICAR_ARROW_LEFT_ANGLE=70
export PICAR_ARROW_RIGHT_ANGLE=110
export PICAR_EVENT_INTERVAL=5
```

Lower `PICAR_EVENT_INTERVAL` means arrows/obstacles are checked more often but inference load goes up.
