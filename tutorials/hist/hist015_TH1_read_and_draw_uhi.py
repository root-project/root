## \file
## \ingroup tutorial_hist
## \notebook -js
## A Simple histogram drawing example.
##
## \macro_image
## \macro_output
## \macro_code
##
## \date July 2025
## \author Wim Lavrijsen, Nursena Bitirgen

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import ROOT
from ROOT import TFile, gROOT

mpl_fig = plt.figure(figsize=(14, 12))
gs = mpl_fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1.5])

# We connect the ROOT file generated in a previous tutorial
File = "py-hsimple.root"
if ROOT.gSystem.AccessPathName(File):
    ROOT.Info("hist015_TH1_read_and_draw.py", File + " does not exist")
    exit()

example = TFile(File)
example.ls()

# Draw histogram hpx in first pad.
hpx = gROOT.FindObject("hpx")

# Get the statistics
entries = int(hpx.GetEntries())
mean = hpx.GetMean()
std_dev = hpx.GetStdDev()
info_text = f"Entries = {entries}\nMean = {mean:.5f}\nStd Dev = {std_dev:.3f}"

ax1 = mpl_fig.add_subplot(gs[0:2, 0])
ax1.set_facecolor("#FFFDD0C8")
hep.histplot(hpx, ax=ax1, histtype="fill", color="#EEAC91", alpha=0.5, edgecolor="blue", linewidth=1.5)
ax1.text(
    0.65,
    0.95,
    info_text,
    transform=ax1.transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
)

# Draw hpx as an interactive lego.
ax2 = mpl_fig.add_subplot(gs[0:2, 1], projection="3d")
ax2.set_facecolor("#FFFDD0C8")
x = np.array([hpx.GetBinCenter(i) for i in range(1, hpx.GetNbinsX() + 1)])
y = np.zeros_like(x)
z = np.zeros_like(x)
dz = hpx.values()
dx = np.full_like(x, hpx.GetBinWidth(1) * 0.9)
dy = np.full_like(x, 0.2)
ax2.bar3d(x, y, z, dx, dy, dz, color="#EEAC91", edgecolor="darkblue", alpha=0.85)
ax2.text2D(
    0.55,
    0.85,
    info_text,
    transform=ax2.transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
)

# Draw hpx with its errors and a marker.
ax3 = mpl_fig.add_subplot(gs[2, :])
ax3.set_facecolor("#FFFDD0C8")
hep.histplot(hpx, ax=ax3, histtype="errorbar", color="darkblue")
ax3.grid(True)
ax3.text(
    0.90,
    0.95,
    info_text,
    transform=ax3.transAxes,
    verticalalignment="top",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
)

mpl_fig.suptitle("Drawing options for one dimensional histograms", fontsize=14, fontweight="bold")
plt.show()

if example.IsOpen():
    example.Close()
