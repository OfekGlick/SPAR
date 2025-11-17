# Rliable Metrics Summary Report

## Aggregate Performance Metrics

| Algorithm | Mean | Median | IQM |
|-----------|------|--------|-----|
| CPPOPID_BAFS_Learned_Budget50pct | 16.209 [13.292, 19.436] | 16.209 [13.292, 19.436] | 16.373 [12.966, 21.171] |
| CPPOPID_BAFS_Learned_Budget80pct | 15.684 [14.292, 17.056] | 15.684 [14.292, 17.056] | 15.718 [14.074, 17.410] |
| PPO_AllObs_Learned | 9.393 [7.427, 11.541] | 9.393 [7.427, 11.541] | 8.228 [6.243, 10.867] |
| PPO_BAFS_Random | 1.433 [1.059, 1.932] | 1.433 [1.059, 1.932] | 1.090 [0.908, 1.433] |
| PPO_BAFS_Learned | 18.924 [16.618, 21.429] | 18.924 [16.618, 21.429] | 19.249 [16.949, 21.451] |
| PPOLag_BAFS_Learned_Budget50pct | 20.162 [17.443, 22.884] | 20.162 [17.443, 22.884] | 19.770 [16.113, 23.148] |
| PPOLag_BAFS_Learned_Budget80pct | 20.703 [18.468, 23.406] | 20.703 [18.468, 23.406] | 19.598 [17.791, 22.502] |

## Notes
- Values shown as: point_estimate [CI_lower, CI_upper]
- Confidence intervals computed using stratified bootstrap (95% CI)
- Analysis based on 140 experimental runs
