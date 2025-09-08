# Accelerator

Utilities for collecting tensor statistics.

## Histogram calibration

`HistogramTensorCollector` can delay histogram creation until a calibration
buffer has gathered enough samples. Pass `calibration_size` to specify the
number of samples per channel to accumulate. The buffer is used to determine
bin ranges using the Freedman–Diaconis rule and is discarded once the
histogram is initialized.
