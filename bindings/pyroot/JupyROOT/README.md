# JupyROOT
A software layer to integrate Jupyter notebooks and ROOT.

## Set a local server up
1. [Install ROOT6](https://root.cern.ch/building-root) (> 6.05)
2. Install Jupyter: pip install jupyter
3. Install metakernel: pip install metakernel
4. Type "root --notebook"

To find the ROOT kernel among the ones automatically detected by Jupyter, just
type:
```
cp -r $ROOTSYS/etc/notebook/kernels/root  ~/.local/share/jupyter/kernels/
```
before starting a ROOTbook make sure you have the ROOT environment properly set
up, i.e. you ran `. $ROOTSYS/bin/thisroot.[c]sh`.

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
