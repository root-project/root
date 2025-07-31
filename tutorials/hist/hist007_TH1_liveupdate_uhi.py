## \file
## \ingroup tutorial_hist
## \notebook -js
## Simple example illustrating how to use the C++ interpreter.
##
## \macro_image
## \macro_code
## \date July 2025
## \author Wim Lavrijsen, Nursena Bitirgen

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from ROOT import TH1F

# Create a new canvas, and customize it.
plt.style.use(hep.style.CMS)
plt.ion()
BINS = 100
RANGE_MIN, RANGE_MAX = -4, 4
K_UPDATE = 500
N_EVENTS = 10000
fig, ax = plt.subplots(figsize=(8, 6), num="The HSUM UHI Example")
# Create some histograms.
total = TH1F("total", "This is the total distribution", BINS, RANGE_MIN, RANGE_MAX)
main = TH1F("main", "Main contributor", 100, -4, 4)
s1 = TH1F("s1", "This is the first signal", 100, -4, 4)
s2 = TH1F("s2", "This is the second signal", 100, -4, 4)
counts = {"total": np.zeros(BINS), "main": np.zeros(BINS), "s1": np.zeros(BINS), "s2": np.zeros(BINS)}
np.random.seed(0)


# Initialize random number generator.
def gauss(loc, scale):
    return np.random.normal(loc, scale)


def landau(loc, scale):
    return np.random.standard_cauchy() * scale + loc


# for speed, bind and cache the Fill member functions
def fill_hist(hist_name, x, weight=1.0):
    if RANGE_MIN <= x < RANGE_MAX:
        idx = int((x - RANGE_MIN) / (RANGE_MAX - RANGE_MIN) * BINS)
        counts[hist_name][idx] += weight


# Fill histograms randomly
for i in range(1, N_EVENTS + 1):
    # Generate random values.
    xmain = gauss(-1, 1.5)
    xs1 = gauss(-0.5, 0.5)
    xs2 = landau(1, 0.15)
    # Fill histograms
    fill_hist("main", xmain)
    fill_hist("s1", xs1, 0.3)
    fill_hist("s2", xs2, 0.2)
    fill_hist("total", xmain)
    fill_hist("total", xs1, 0.3)
    fill_hist("total", xs2, 0.2)
    total[...] = counts["total"]
    main[...] = counts["main"]
    s1[...] = counts["s1"]
    s2[...] = counts["s2"]
    # Update display every kUPDATE events.
    if i % K_UPDATE == 0:
        ax.cla()
        entries = total.GetEntries()
        mean = total.GetMean()
        stddev = total.GetStdDev()
        stats_text = f"Entries = {entries}\nMean = {mean:.2f}\nStd Dev = {stddev:.2f}"
        hep.histplot(main, histtype="fill", color="gray", alpha=0.5, edgecolor="blue", linewidth=1.5, ax=ax)
        hep.histplot(total, histtype="errorbar", color="black", ecolor="blue", linewidth=2, ax=ax)
        hep.histplot(s1, histtype="errorbar", color="blue", alpha=0.7, ecolor="blue", linewidth=2, marker="+", ax=ax)
        hep.histplot(s2, histtype="errorbar", color="blue", alpha=0.7, ecolor="blue", linewidth=2, marker="+", ax=ax)
        ax.set_title("This is the total distribution", pad=20, fontsize=14, loc="center")
        ax.text(
            0.95,
            0.90,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.9),
        )
        # Plot formatting
        ax.set_xlim(RANGE_MIN, RANGE_MAX)
        ax.set_ylim(0, max(counts["total"]) * 1.2)
        plt.pause(0.01)
# Done, finalized and trigger an update.
plt.grid(True)
plt.ioff()
plt.show()
