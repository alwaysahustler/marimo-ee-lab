
# PID Controller Notebook

This folder contains a Marimo notebook for control-systems analysis of a second-order plant using P, PI, and PID control.

## Notebook

- `PID_controller.py`: interactive notebook for tuning controller gains and observing closed-loop response

## Plant Model

The notebook studies the plant

$$
G(s) = \frac{1}{s^2 + 2s + 3}
$$

## Controller Modes

- P: `Ki = 0`, `Kd = 0`
- PI: `Kd = 0`
- PID: all gains active

Inactive gains are disabled in the UI so the notebook stays consistent with the selected controller type.

## Output

The notebook displays:

- Closed-loop step response
- Error response
- Standard performance metrics such as stability, steady-state error, overshoot, rise time, and settling time

## Run

From the repository root, open the notebook with Marimo:

```bash
marimo edit control-system/PID/PID_controller.py
```

Or run it directly:

```bash
marimo run control-system/PID/PID_controller.py
```

