# ROOTaaS
A tool to integrate the iPython notebooks, ROOT and jsROOT. This tool is an add-on to pyROOT and is still in the prototype phase.

## Set the server up
1. Get ROOT6, set it up.
2. Get iPython, at least version 3. 
3. Clone the repository
4. Type "ipython notebook"

## Example usage
```python
from ROOTaaS.iPyROOT import ROOT
h = ROOT.TH1F("h","iPython Histo;X;Y",64,-4,4)
h.FillRandom("gaus")
h.Draw()
```

## Magics and interaction with C++
 * %%cpp for marking a cell for C++
 * %%dcl for marking a cell for declaring C++ (e.g. for functions)
 * ROOT.toCpp() to move from Python to C++
 * toPython() to move from C++ to Python
 * .decl for marking a cell for declaring C++ (e.g. for functions) in C++ mode (as %%dcl)
