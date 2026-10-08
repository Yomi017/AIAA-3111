# Predeclared mechanism study

Written before the new development run. Theme and primary task remain normal-only NASA SMAP/MSL anomaly detection. Earlier tests have been exposed, so this study can improve protocol discipline but cannot restore an independent test.

H1: A late normal calibration segment underrepresents legitimate regimes. Test a conservative normal envelope: maximum of the 99.5th percentile in normal selection and calibration. This only changes the threshold, never the subspace score. Expected signature: lower false-positive rate, possibly lower recall. This is not an FPR guarantee under temporal distribution shift.

H2: A single threshold does not fit every normal level. Partition windows by normal-fit median-level quartiles, then calibrate each bin at 99.5% using normal calibration. Bins with fewer than 30 calibration scores fall back to the global threshold. Expected signature: fewer false positives at difficult normal levels. Rare or new levels may still fail. No test-based binning, label calibration, or test-time updating.

H3: High-dimensional waveforms are sensitive to trajectory variation. Use eight robust trajectory descriptors (median, IQR, last level, drift, median and 90th-percentile absolute step, OLS slope, half-window mean contrast), fit-only robust scaling, and the same stable novelty LOF with k=50. Expected signature: more stable cross-channel behavior, with possible loss of subtle shape anomalies. It is a single representation, not a score ensemble.

Reference: the existing PCA95 whitened stable LOF50, span1, normal calibration99.5. Window64 and stride2 are held fixed. No grid or additional method will be added after confirmation.

Evaluate all four arms on the 22 development channels. Select by equal-weight mean of SMAP/MSL micro point F1, freeze the exact config and evidence hash, then evaluate the four fixed arms once on the 58 confirmation channels as declared mechanism ablations. Only the development-selected arm is prospective for this iteration. The historical candidate remains exploratory. Do not switch to the highest confirmation score. Keep every arm, including failures, and report channel bootstrap uncertainty and event recall.

Existing causal multi-horizon GRU, USAD adaptation, temporal PCA and nearest-subsequence experiments provide method comparisons; do not relabel them as exact paper replications. Supervised HGB/logistic grouped resampling is a separate labeled-source task.
