# Author: Stephanie Kwan CERN 11/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
\pythondoc THistPainter

### Keeping the canvas open after drawing in Python
- If the call to TH1.Draw is a top-level statement (i.e. the histogram remains in scope at the end of the script), e.g. 
\code{.py}
# short example makeAndDrawHisto.py: initialize a histogram and draw it on a canvas
test_histo = ROOT.TH1D("test_histo", "Histogram to draw", 200, -5, 5)
test_histo.FillRandom("gaus", 1000)
test_canvas = ROOT.TCanvas("test_canv", "test_canv", 900, 700)
test_histo.Draw()
\endcode
It is sufficient to call Python with the flag `-i` to keep the TBrowser open:
\code{.sh}
# -i flag keeps the TBrowser open with the TCanvas on it
python -i makeAndDrawHisto.py
\endcode

- If the call to TH1.Draw is not at top-level, both the TCanvas and TH1 objects need to remain in scope. One way to accomplish this is with ROOT.SetOwnership, as in this example:
\code{.py}
# contents of short example makeAndDrawHistoInMain.py: 
def main():
    '''
    Initialize a histogram and draw it in a non-top level function, using ROOT.SetOwnership to keep the canvas and histogram open after execution
    '''
    test_histo = ROOT.TH1D("test_histo", "Histogram to draw", 200, -5, 5)
    test_histo.FillRandom("gaus", 1000)

    test_canv = ROOT.TCanvas("test_canv", "test_canv", 900, 700)
    ROOT.SetOwnership(test_canv, False)
    ROOT.SetOwnership(test_histo, False)
    test_canv.SaveAs("myDrawExample.png")

if __name__ == '__main__':
    main()
\endcode
To keep the TBrowser open, the Python script should be run with the `-i` flag:
\code{.sh}
python -i makeAndDrawHistoInMain.py
\endcode

\endpythondoc
"""

from . import pythonization
