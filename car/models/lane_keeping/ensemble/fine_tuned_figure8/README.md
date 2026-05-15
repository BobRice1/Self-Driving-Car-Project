# Figure-8 Fine-Tuned Lane Models

Fine-tuned from the existing right-weighted lane checkpoints using the additional data in `car/data/lane_keeping/figure8`.

## Data

- Old training rows used: 4500
- Figure-8 training rows used: 177
- Old validation rows used: 2500
- Figure-8 validation rows used: 44
- Figure-8 sample weight: 12
- Steering range: 50 to 135
- Straight angle used during training metadata: 94

## Outputs

- `lane_nvidia.tflite`
- `lane_mobilenetv3_large.tflite`
- `lane_nvidia.onnx`
- `lane_mobilenetv3_large.onnx`

Detailed checkpoints, logs, and split CSVs are also kept in the `nvidia` and `mobilenet` subdirectories.

## Best Validation Results

| Model | Best epoch | Old MAE | Figure-8 MAE | Score |
| --- | ---: | ---: | ---: | ---: |
| Nvidia | 3 | 8.44 | 11.57 | 10.48 |
| MobileNetV3 Large | 6 | 5.42 | 8.60 | 7.49 |
