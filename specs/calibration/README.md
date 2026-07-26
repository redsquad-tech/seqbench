# Calibration

`full_v1.yaml` intentionally remains uncalibrated until `v1.json` is generated
from weak and strong reference runs:

```bash
seqbench calibrate \
  --weak runs/random \
  --weak runs/memorizer \
  --strong runs/reference \
  --output specs/calibration/v1.json
```
