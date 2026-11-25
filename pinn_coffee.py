# PINN example: a cup of coffee that cools down.
# Physics: Newton's law of cooling   dT/dt = -k * (T - T_room)
# Data: 5 temperature measurements in the first 10 minutes.
# Goal: predict the temperature for the full 60 minutes.
#
# Run:  python pinn_coffee.py
# Needs: torch, matplotlib (matplotlib is optional, for the plot)

import time
import torch

torch.manual_seed(0)

# ---- 1. The problem ----
T_ROOM = 22.0   # room temperature, deg C
T_START = 90.0  # coffee temperature at t = 0, deg C
K = 0.05        # cooling rate, 1/minute
T_END = 60.0    # minutes


def true_temperature(t):
    return T_ROOM + (T_START - T_ROOM) * torch.exp(-K * t)


# ---- 2. The data: only 5 measurements, all in the first 10 minutes ----
t_data = torch.tensor([[0.0], [2.5], [5.0], [7.5], [10.0]])
T_data = true_temperature(t_data)

# ---- 3. The physics points: 50 times where we have NO measurement ----
t_physics = torch.linspace(0.0, T_END, 50).reshape(-1, 1).requires_grad_(True)


# ---- 4. The network: time in, temperature out ----
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, t):
        # scale the input to 0..1 and the output to degrees C
        return 100.0 * self.layers(t / T_END)


def train(use_physics, steps=5000):
    torch.manual_seed(0)
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for step in range(steps + 1):
        optimizer.zero_grad()

        # Loss 1: agree with the measurements
        loss_data = torch.mean((net(t_data) - T_data) ** 2)

        # Loss 2: agree with the physics
        loss_physics = torch.tensor(0.0)
        if use_physics:
            T = net(t_physics)
            dT_dt = torch.autograd.grad(T, t_physics, torch.ones_like(T),
                                        create_graph=True)[0]
            residual = dT_dt + K * (T - T_ROOM)   # zero if the law is obeyed
            loss_physics = torch.mean(residual ** 2)

        loss = loss_data + 100.0 * loss_physics
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"  step {step:5d}  data loss {loss_data.item():10.4f}"
                  f"  physics loss {loss_physics.item():10.6f}")
    return net


if __name__ == "__main__":
    print("Plain network (data only)")
    start = time.time()
    plain = train(use_physics=False)
    print(f"  time: {time.time() - start:.1f} s")

    print("PINN (data + physics)")
    start = time.time()
    pinn = train(use_physics=True)
    print(f"  time: {time.time() - start:.1f} s")

    # ---- 5. Compare ----
    t_test = torch.linspace(0.0, T_END, 200).reshape(-1, 1)
    truth = true_temperature(t_test)
    with torch.no_grad():
        p_plain, p_pinn = plain(t_test), pinn(t_test)

    print()
    print(f"True temperature at 60 min:  {truth[-1].item():6.1f} C")
    print(f"Plain network at 60 min:     {p_plain[-1].item():6.1f} C")
    print(f"PINN at 60 min:              {p_pinn[-1].item():6.1f} C")
    print(f"Largest error, plain network: {(p_plain - truth).abs().max().item():6.1f} C")
    print(f"Largest error, PINN:          {(p_pinn - truth).abs().max().item():6.2f} C")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.5))
        plt.plot(t_test, truth, "k--", label="True temperature")
        plt.plot(t_test, p_plain, label="Plain network (data only)")
        plt.plot(t_test, p_pinn, label="PINN (data + physics)")
        plt.scatter(t_data, T_data, color="k", zorder=5, label="5 measurements")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Temperature (deg C)")
        plt.title("Coffee that cools down: plain network and PINN")
        plt.legend()
        plt.tight_layout()
        plt.savefig("pinn_coffee.png", dpi=150)
        print("Plot saved: pinn_coffee.png")
    except ImportError:
        pass
