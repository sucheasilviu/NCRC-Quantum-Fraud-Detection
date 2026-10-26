# Noise types and uneven noise (src/noise_types.py)

Exact scores, 500-row fraud-rich test set, 5 seeds; uneven kinds: 5 random devices per seed (25 runs).
Logistic regression (no gates): AUPRC 0.959 ± 0.002 under every noise.

| model | noise | runs | AUPRC p=0 | mean p=0.1 | worst p=0.1 | mean p=0.2 | worst p=0.2 | worst ρ p=0.2 |
|---|---|---|---|---|---|---|---|---|
| baseline | depolarizing | 5 | 0.947 | 0.765 | 0.110 | 0.764 | 0.107 | -0.657 |
| baseline | amplitude_damping | 5 | 0.947 | 0.890 | 0.832 | 0.466 | 0.154 | -0.815 |
| baseline | phase_damping | 5 | 0.947 | 0.940 | 0.935 | 0.778 | 0.170 | -0.113 |
| baseline | bit_flip | 5 | 0.947 | 0.873 | 0.669 | 0.877 | 0.715 | -0.027 |
| baseline | hetero_depolarizing | 25 | 0.947 | 0.765 | 0.108 | 0.763 | 0.107 | -0.694 |
| baseline | hetero_mixed | 25 | 0.947 | 0.879 | 0.134 | 0.564 | 0.107 | -0.931 |
| ncrc_1q_L2_pruned | depolarizing | 5 | 0.956 | 0.956 | 0.953 | 0.956 | 0.953 | 1.000 |
| ncrc_1q_L2_pruned | amplitude_damping | 5 | 0.956 | 0.956 | 0.953 | 0.956 | 0.953 | 0.993 |
| ncrc_1q_L2_pruned | phase_damping | 5 | 0.956 | 0.956 | 0.953 | 0.956 | 0.953 | 0.999 |
| ncrc_1q_L2_pruned | bit_flip | 5 | 0.956 | 0.956 | 0.952 | 0.956 | 0.952 | 0.990 |
| ncrc_1q_L2_pruned | hetero_depolarizing | 25 | 0.956 | 0.956 | 0.953 | 0.956 | 0.953 | 1.000 |
| ncrc_1q_L2_pruned | hetero_mixed | 25 | 0.956 | 0.956 | 0.952 | 0.955 | 0.930 | 0.977 |
| ncrc_4q_L4_cnot | depolarizing | 5 | 0.955 | 0.956 | 0.949 | 0.956 | 0.947 | 0.955 |
| ncrc_4q_L4_cnot | amplitude_damping | 5 | 0.955 | 0.956 | 0.947 | 0.905 | 0.873 | 0.159 |
| ncrc_4q_L4_cnot | phase_damping | 5 | 0.955 | 0.956 | 0.949 | 0.956 | 0.950 | 0.994 |
| ncrc_4q_L4_cnot | bit_flip | 5 | 0.955 | 0.955 | 0.943 | 0.935 | 0.922 | 0.749 |
| ncrc_4q_L4_cnot | hetero_depolarizing | 25 | 0.955 | 0.956 | 0.948 | 0.955 | 0.946 | 0.943 |
| ncrc_4q_L4_cnot | hetero_mixed | 25 | 0.955 | 0.956 | 0.945 | 0.923 | 0.691 | -0.205 |
| ncrc_4q_L4 | depolarizing | 5 | 0.957 | 0.957 | 0.949 | 0.957 | 0.949 | 1.000 |
| ncrc_4q_L4 | amplitude_damping | 5 | 0.957 | 0.956 | 0.948 | 0.955 | 0.940 | 0.984 |
| ncrc_4q_L4 | phase_damping | 5 | 0.957 | 0.957 | 0.949 | 0.957 | 0.949 | 0.995 |
| ncrc_4q_L4 | bit_flip | 5 | 0.957 | 0.956 | 0.948 | 0.955 | 0.946 | 0.942 |
| ncrc_4q_L4 | hetero_depolarizing | 25 | 0.957 | 0.957 | 0.948 | 0.956 | 0.948 | 0.954 |
| ncrc_4q_L4 | hetero_mixed | 25 | 0.957 | 0.957 | 0.948 | 0.954 | 0.931 | 0.932 |
