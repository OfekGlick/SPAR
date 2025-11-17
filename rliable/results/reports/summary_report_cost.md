# Rliable Metrics Summary Report

## Aggregate Performance Metrics

| Algorithm | Mean | Median | IQM |
|-----------|------|--------|-----|
| CPPOPID_BAFS_Learned_Budget50pct | 409.986 [322.256, 496.879] | 409.986 [322.256, 496.879] | 457.188 [328.295, 524.854] |
| CPPOPID_BAFS_Learned_Budget80pct | 756.070 [699.924, 812.028] | 756.070 [699.924, 812.028] | 764.916 [692.940, 817.744] |
| PPO_AllObs_Learned | 1500.000 [1500.000, 1500.000] | 1500.000 [1500.000, 1500.000] | 1500.000 [1500.000, 1500.000] |
| PPO_BAFS_Random | 750.198 [749.242, 751.150] | 750.198 [749.242, 751.150] | 750.338 [748.834, 751.626] |
| PPO_BAFS_Learned | 796.458 [714.740, 875.428] | 796.458 [714.740, 875.428] | 762.096 [704.839, 853.712] |
| PPOLag_BAFS_Learned_Budget50pct | 600.375 [536.906, 661.903] | 600.375 [536.906, 661.903] | 606.082 [533.294, 687.049] |
| PPOLag_BAFS_Learned_Budget80pct | 765.260 [701.806, 826.506] | 765.260 [701.806, 826.506] | 799.990 [710.441, 854.942] |

## Notes
- Values shown as: point_estimate [CI_lower, CI_upper]
- Confidence intervals computed using stratified bootstrap (95% CI)
- Analysis based on 140 experimental runs
