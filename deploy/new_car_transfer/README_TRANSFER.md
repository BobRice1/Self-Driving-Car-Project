# New car transfer package

Pi-ready model folder:
- Copy `autopilot/models/BL` into `/home/pi/autopilot/autopilot/models/BL` on the car.
- Run from `/home/pi/autopilot` with: `python3 -u run.py --model BL --mode drive --duration 60 --max_speed 25`

Current test setup:
- `USE_ARROW_TURNS = False` in `model.py` for lane-only testing.
- `INVERT_LANE_STEERING = True`.
- `CROP_RATIO = 0.25` runtime look-ahead test.

Source artifacts:
- `source_artifacts/car/checkpoints` contains latest lane/arrow checkpoints and TFLite files.
- `source_artifacts/car/inference` contains the modular inference wrappers.
