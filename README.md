# MindGap: AI Explained

Code for the videos on the MindGap YouTube channel. Each video with code has its own folder.

| Video | Folder | Watch |
|---|---|---|
| What is a PINN? Physics-Informed Neural Networks Explained | [`pinn-coffee`](pinn-coffee) | [YouTube](https://youtu.be/tJ4D3KqbQAI) |

## What is a PINN? (`pinn-coffee`)

A physics-informed neural network (PINN) is a neural network that learns from data and must also obey a law of physics. The law is written as an equation.

This example trains two networks on the same problem and compares them:

- A plain network, which learns only from the data.
- A PINN, which learns from the data and from Newton's law of cooling.

### The problem

A cup of coffee starts at 90 °C in a room at 22 °C. We have only 5 temperature measurements, all in the first 10 minutes. We want the temperature for the full 60 minutes.

Newton's law of cooling:

```
dT/dt = -k * (T - T_room)
```

### Run it

You need Python 3.9 or later.

```bash
pip install torch matplotlib
python pinn-coffee/pinn_coffee.py
```

The training is 5,000 steps for each network. It takes a few seconds on a CPU. No graphics card is necessary.

### Example output

These numbers are from one run. Your numbers can be a little different.

| | Temperature at 60 minutes |
|---|---|
| True value | 25.4 °C |
| Plain network (data only) | -47.0 °C |
| PINN (data + physics) | 25.4 °C |

The largest error of the PINN in the full hour was 0.03 °C. The script also saves a plot, `pinn_coffee.png`.

### How it works

The PINN has two losses, and the training makes their sum as small as possible.

1. **Data loss:** does the network agree with the 5 measurements?
2. **Physics loss:** does the network obey the equation? The script checks this at 50 points in time where there are no measurements.

These are the lines that make the network a PINN:

```python
T = net(t_physics)
dT_dt = torch.autograd.grad(T, t_physics, torch.ones_like(T),
                            create_graph=True)[0]
residual = dT_dt + K * (T - T_ROOM)   # zero if the law is obeyed
loss_physics = torch.mean(residual ** 2)
```

Automatic differentiation gives the rate of change `dT_dt` directly from the network. If the network obeys the law, the residual is zero.

### Things to try

- Change `K`, the cooling rate, and run the script again.
- Change the weight of the physics loss (the `100.0` in the training loop).
- Use fewer measurements, or move them to a different time.
- Make `K` a parameter that the network learns. This is an inverse problem.

### Limits

- The physics must be correct. If the equation is wrong, the network learns the wrong law.
- The two losses compete, so you must balance them.
- For many problems, a classical numerical solver is faster and more accurate.

### Reference

Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

## About

MindGap makes videos about AI, agents, and AI-assisted building: [mindgap.tech](https://mindgap.tech)
