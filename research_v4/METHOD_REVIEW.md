# Primary-source method review

Read on 2026-10-06; source bytes, retrieval URLs, timestamps and SHA-256 are in sources.json. Earlier research_v2 snapshots and licenses are retained. No GitHub repository was cloned for this review.

## Predictive family: Telemanom

Hundman et al., Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding, KDD 2018. Paper https://arxiv.org/abs/1802.04431 ; official https://github.com/khundman/telemanom . Read the abstract, README dataset section, config, channel shaping code and errors implementation.

Mechanism: normal LSTM predicts telemetry using its history and command indicators; prediction errors are smoothed and thresholded by an unlabeled statistical objective with sequence buffering/pruning. Official config uses history250, ten future predictions, layers80/80, error_buffer100. This project uses history64 GRU32, one-step or direct16-horizon predictions and a separate normal99.5 threshold. It does not reproduce Telemanom's exact model or event rules. The public values have already been scaled by TEST min/max (README). Local fit-only scaling cannot undo this inherited exposure. Command indicators are module-local anonymized inputs, not synchronized measurements from different channels.

## Adversarial reconstruction family: USAD

Audibert, Michiardi, Guyard, Marti and Zuluaga, USAD, KDD 2020. https://doi.org/10.1145/3394486.3403392 ; https://github.com/manigalati/usad . Read official README and complete usad.py.

Mechanism: shared encoder and two decoders; losses at epoch n are L1=(1/n)||w-AE1(w)||²+(1-1/n)||w-AE2(AE1(w))||² and L2=(1/n)||w-AE2(w)||²-(1-1/n)||w-AE2(AE1(w))||². The second optimizer opposes cascade reconstruction. Official testing weights first reconstruction and cascade reconstruction equally. Local reconstruction_v2.py is an adaptation with linear outputs for standardized values, normal checkpoint selection, and a target-only window64. It is not the original multivariate benchmark protocol. Local scores are recalibrated normally and unadjusted.

## Attention and self-conditioning family: TranAD

Tuli, Casale and Jennings, TranAD, PVLDB15(6),1201-1214,2022. https://arxiv.org/abs/2201.07284 ; https://github.com/imperial-qore/TranAD . Read abstract, README, TranAD class in models.py and current src/pot.py.

Mechanism: attention encoders with positional information, dual decoder phases, second phase self-conditioned on first-phase squared residuals. Abstract also describes adversarial training and MAML. Read class has window10 and sigmoid output; project does not train TranAD. Official pot.py has adjust_predicts(score,label), which fills an entire real anomaly after a hit, including backfilling; bf_search also searches labeled best F1. Presence of those utilities does not prove every published number uses every utility, but is sufficient to reject naive comparisons of README scores against this project's raw point F1. No imported score table is included in local comparisons.

## Local density and distribution adaptation: TSB-AD

Liu and Paparrizos, The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark, NeurIPS2024. https://github.com/TheDatumOrg/TSB-AD . Read README plus models/LOF.py and models/M2N2.py. Local density originates in Breunig et al., LOF: Identifying Density-Based Local Outliers, SIGMOD2000, https://doi.org/10.1145/342009.335388 .

LOF compares neighborhood reachability density. TSB-AD wraps sliding windows and sklearn LOF, with normalizations and score-edge padding. Its decision_function can normalize evaluation windows from their own statistics; its padding is centered. Our trailing windows assign a score at their right endpoint, use fit-only normalization, no padding, and a numerically regularized novelty density against normal references. It is inspired by the family, not an exact TSB-AD reproduction.

M2N2 uses a normal autoencoder, an EMA detrender and online learning from samples predicted normal. Read infer_online updates trend statistics and masks presumed anomalies before gradient adaptation; fit derives a training quantile threshold. This suggests legitimate distribution shifts can produce false positives, but updating a spacecraft detector on its own decisions risks absorbing a real persistent fault. The new study therefore tests fixed normal coverage and fixed robust descriptors rather than adding unvalidated online adaptation. M2N2 was reviewed, not trained.

## Protocol implication

At least three mechanistically different families were read from primary papers or official implementations. All local comparisons share t>=64, inclusive labels, no point adjustment. Score magnitude is not compared between algorithms. Chronological normal model selection and calibration are distinct from anomaly-labeled channel development. Cross-channel supervised HGB/logistic uses nested channel groups and must be reported as a separate labeled-source problem. Confirmation has been exposed through prior versions and parameter-grid evaluation, so intervals and gains remain exploratory.
