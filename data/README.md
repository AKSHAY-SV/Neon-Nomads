# Data directory

This folder is intentionally empty of real data in this repository — the
actual CPRI test-bench dataset is not included (no synthetic/fake data has
been generated as a substitute). Place the two required files here before
running `src/main.py`:

## `training_data.csv`

Required columns:

```
Test_ID
Applied_Voltage_kV
Load_Current_A
Ambient_Temperature_C
Test_Duration_min
Sensor_S1
Sensor_S2
Sensor_S3
Sensor_S4
Reference_Parameter
Validity_Label
```

## `test_data.csv`

Required columns:

```
Test_ID
Applied_Voltage_kV
Load_Current_A
Ambient_Temperature_C
Test_Duration_min
Sensor_S1
Sensor_S2
Sensor_S3
Sensor_S4
```

`Validity_Label` values are expected to be `"Valid"` / `"Invalid"`.

If either file is missing, `src/main.py` exits with a clear error message
instead of failing silently or fabricating data.
