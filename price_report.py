import math
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from options import black_scholes_call, monte_carlo_call, asian_call, barrier_call, geometric_asian_closed_form

# Config
SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
VOLATILITY = 0.2
EXPIRY = 1.0
PATHS = 200_000
STEPS = 50

# Dark palette
BG = "#12151c"
PANEL = "#181c26"
TEXT = "#d6dae3"
GRID = "#2b3140"
UP = "#ff5c72"
DOWN = "#5a6274"
BLUE = "#4fc3f7"
PURPLE = "#b388ff"
GOLD = "#ffca4f"


def journey(rng, vol=VOLATILITY):
    # One simulated share price path, step by step
    dt = EXPIRY / STEPS
    drift = (RATE - vol * vol / 2.0) * dt
    step = vol * math.sqrt(dt)
    price = SPOT
    out = [price]
    for _ in range(STEPS):
        u1, u2 = rng.random(), rng.random()
        while u1 <= 0.0:
            u1 = rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        price *= math.exp(drift + step * z)
        out.append(price)
    return out


exact = black_scholes_call(SPOT, STRIKE, RATE, VOLATILITY, EXPIRY)
print(f"\nspot {SPOT:.0f} | strike {STRIKE:.0f} | vol {VOLATILITY:.0%} | {PATHS:,} paths\n")

print("convergence")
for n in (1_000, 100_000, 1_000_000):
    mc = monte_carlo_call(SPOT, STRIKE, RATE, VOLATILITY, EXPIRY, n, seed=1, antithetic=True)
    print(f"  {n:>9,}  {mc:7.4f}  exact {exact:.4f}  err {abs(mc - exact):.4f}")

print("\nexotics")
geo_exact = geometric_asian_closed_form(SPOT, STRIKE, RATE, VOLATILITY, EXPIRY, STEPS)
geo = asian_call(SPOT, STRIKE, RATE, VOLATILITY, EXPIRY, PATHS, steps=STEPS, seed=4, geometric=True)
ari = asian_call(SPOT, STRIKE, RATE, VOLATILITY, EXPIRY, PATHS, steps=STEPS, seed=4)
bar = barrier_call(SPOT, STRIKE, 130.0, RATE, VOLATILITY, EXPIRY, PATHS, steps=STEPS, seed=5)
print(f"ordinary          {exact:7.4f}")
print(f"Asian arithmetic  {ari:7.4f}  no formula exists")
print(f"Asian geometric   {geo:7.4f}  exact {geo_exact:.4f}")
print(f"barrier at 130    {bar:7.4f}  {bar / exact:.0%} of ordinary")


def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=11, pad=10)
    ax.set_xlabel("years", color=TEXT, fontsize=9)
    ax.set_ylabel("share price", color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(color=GRID, alpha=0.6, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)


times = [i * EXPIRY / STEPS for i in range(STEPS + 1)]
fig, axes = plt.subplots(3, 2, figsize=(14, 13), facecolor=BG)
fig.suptitle("Monte Carlo option pricing on Forge", fontsize=16, color=TEXT, y=0.985)

# paths coloured by whether they pay out
ax = axes[0][0]
rng = random.Random(9)
paid = 0
for _ in range(80):
    j = journey(rng)
    win = j[-1] > STRIKE
    paid += win
    ax.plot(times, j, lw=0.8, alpha=0.8 if win else 0.5, color=UP if win else DOWN)
ax.axhline(STRIKE, color=TEXT, ls="--", lw=1.2, label=f"strike ({STRIKE:.0f})")
style(ax, f"80 simulated futures, {paid} finish in the money")

# how far the paths spread as time passes
ax = axes[0][1]
rng = random.Random(21)
paths = [journey(rng) for _ in range(4000)]
bands = []
for i in range(STEPS + 1):
    col = sorted(p[i] for p in paths)
    n = len(col)
    bands.append((col[n // 20], col[n // 4], col[n // 2], col[3 * n // 4], col[19 * n // 20]))
ax.fill_between(times, [b[0] for b in bands], [b[4] for b in bands], color=BLUE, alpha=0.15, label="middle 90%")
ax.fill_between(times, [b[1] for b in bands], [b[3] for b in bands], color=BLUE, alpha=0.32, label="middle 50%")
ax.plot(times, [b[2] for b in bands], color=TEXT, lw=1.8, label="median")
ax.axhline(STRIKE, color=UP, ls="--", lw=1.2)
style(ax, "Uncertainty grows with time, which is why options cost money")

# barrier knockouts
ax = axes[1][0]
rng = random.Random(33)
knocked = 0
for _ in range(80):
    j = journey(rng)
    out = max(j) >= 130.0
    knocked += out
    ax.plot(times, j, lw=0.9 if out else 0.8, alpha=0.85 if out else 0.4, color=UP if out else DOWN)
ax.axhline(130.0, color=UP, lw=1.8, label="barrier (130)")
ax.axhline(STRIKE, color=TEXT, ls="--", lw=1.0, label=f"strike ({STRIKE:.0f})")
style(ax, f"Barrier option: {knocked} of 80 paths knocked out")

# Asian option pays on the running average
ax = axes[1][1]
rng = random.Random(44)
for i, colour in enumerate((UP, BLUE, PURPLE)):
    j = journey(rng)
    running = []
    total = 0.0
    for step, price in enumerate(j[1:], start=1):
        total += price
        running.append(total / step)
    ax.plot(times, j, lw=0.9, alpha=0.35, color=colour)
    ax.plot(times[1:], running, lw=2.2, color=colour, label=f"average {i + 1}")
ax.axhline(STRIKE, color=TEXT, ls="--", lw=1.0)
style(ax, "Asian option pays on the average, the thick lines")

# low against high volatility
ax = axes[2][0]
for vol, colour in ((0.10, BLUE), (0.40, UP)):
    rng = random.Random(55)
    for _ in range(30):
        ax.plot(times, journey(rng, vol), lw=0.8, alpha=0.45, color=colour)
    ax.plot([], [], color=colour, lw=2, label=f"{vol:.0%} volatility")
ax.axhline(STRIKE, color=TEXT, ls="--", lw=1.0)
style(ax, "More volatility means more paths finishing far from the strike")

# the payout is the gap above the strike
ax = axes[2][1]
rng = random.Random(66)
for _ in range(80):
    j = journey(rng)
    win = j[-1] > STRIKE
    ax.plot(times, j, lw=0.8, alpha=0.8 if win else 0.35, color=GOLD if win else DOWN)
    if win:
        ax.plot([EXPIRY, EXPIRY], [STRIKE, j[-1]], lw=2.0, alpha=0.7, color=GOLD)
ax.axhline(STRIKE, color=TEXT, ls="--", lw=1.2, label="payout measured from here")
style(ax, "The payout is the gap above the strike at expiry")

fig.tight_layout(rect=[0, 0, 1, 0.975])
fig.savefig("prices.png", dpi=140, facecolor=BG)
print("\nprices.png written\n")