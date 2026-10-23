# BRFSS Health Regression Analysis
Multiple linear regression predicting self-rated health from education, income, and BMI, using CDC Behavioral Risk Factor Surveillance System (BRFSS) survey data from 2011–2014.
The full run covers 1,582,924 valid records and is computed in a single streaming pass. The data is processed in chunks without loading the full dataset into memory.

## Results
- Coefficients β = [3.0827, −0.1516 (education), −0.1443 (income), +0.0368 (BMI)]
- Adjusted R² = 0.1994

## My contribution
- Wrote a comprehensive test suite (22 tests) covering different data shapes and subsets, streaming/chunking equivalence, and edge cases: bad columns, missing files, empty data, singular/collinear inputs, boundary sample sizes, constant-y, invalid field codes, and require-all on/off behavior
- Verified coefficients and adjusted R² against NumPy's solver
- Ran and verified the full-dataset result

## Files
- `data_processing.py` — streaming data loading / processing
- `regression_part2.py` — regression implementation
- `test.py` — my comprehensive test suite (22 tests)

## Tech
Python · NumPy (verification) · streaming / chunked processing · test-driven verification

## Data
BRFSS data (CDC, 2011–2014) isn't included in this repo due to its size, but it can be downloaded from the CDC BRFSS website (LLCP2011ASC–LLCP2014ASC).
