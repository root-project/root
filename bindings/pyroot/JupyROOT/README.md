# JupyROOT
A tool to integrate the Jupyter notebooks and ROOT.

## Set a local server up
1. Install ROOT6 (> 6.05)
2. Install Jupyter: pip install jupyter
3. Install metakernel: pip install metakernel
4. Type "root --notebook"

## Example usage
```python
import ROOT
c = ROOT.TCanvas()
h = ROOT.TH1F("h","iPython Histo;X;Y",64,-4,4)
h.FillRandom("gaus")
h.Draw()
c.Draw()
```

## Magics and interaction with C++
 * ROOT C++ Kernel
 * %%cpp for marking a cell for C++
