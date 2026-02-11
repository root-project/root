# Author: Sergey Linev GSI  01/2025

################################################################################
# Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
\pythondoc TCanvas

The TCanvas class is used to create the canvas on which graphical objects such as histograms can be drawn. The following
is a simple example of typical usage:

\code{.py}
import ROOT
import numpy


def plot():
    h = ROOT.TH1D("h", "h", 100, -5, 5)
    h.Fill(numpy.random.normal(size=1000))

    c = ROOT.TCanvas()
    h.Draw()
    c.Draw(block=True)


if __name__ == "__main__":
    print("Before plot function")
    plot()
    print("After plot function")
\endcode

Note the optional argument `block` passed to the `Draw` method of the canvas. If set to `True`, it will block the script
execution and run the ROOT graphics event loop until the <space> key is pressed. This allows interacting with the
canvas and its content until necessary, then move on with the rest of the script.

Another relevant use case is drawing live updates on a canvas, shown in the example below. In this case, the canvas is
setup by first creating the object to be drawn and drawing it once. Then, the object is updated in a for loop, which
could represent for example an incoming stream of data with which the histogram should be filled. Each time the plot
should be updated, the `ModifiedUpdate` function should be called. This will immediately show the new contents on the
plot. Finally, the canvas is drawn again with `Draw(block=True)` at the end of the loop so that it stays visible and can
be interacted with.

\code{.py}
import ROOT
import numpy


def live_update():
    c = ROOT.TCanvas()
    h = ROOT.TH1D("h", "h", 100, -5, 5)

    h.Draw()
    for _ in range(100):
        h.Fill(numpy.random.normal(size=10))
        c.ModifiedUpdate()
    c.Draw(block=True)

if __name__ == "__main__":
    print("Before plot function")
    live_update()
    print("After plot function")
\endcode

\endpythondoc
"""

from . import _run_root_event_loop, pythonization


def _TCanvas_Update(self, block=False):
    """
    Updates the canvas.
    Also blocks script execution and runs the ROOT graphics event loop until the <space> key is pressed,
    but only if the following conditions are met:
    * The `block` optional argument is set to `True`.
    * ROOT graphics are enabled, i.e. `ROOT.gROOT.IsBatch() == False`.
    * The script is not running in ipython notebooks.
    """

    self._Update()

    # run loop if block flag is set
    if block:
        _run_root_event_loop()


def _TCanvas_Draw(self, option: str = "", block: bool = False):
    """
    Draw the canvas.
    Also blocks script execution and runs the ROOT graphics event loop until the <space> is pressed,
    but only if the following conditions are met:
    * The `block` optional argument is set to `True`.
    * ROOT graphics are enabled, i.e. `ROOT.gROOT.IsBatch() == False`.
    * The script is not running in ipython notebooks.
    """

    self._Draw(option)

    # run loop if block flag is set
    if block:
        self._Update()
        _run_root_event_loop()


@pythonization("TCanvas")
def pythonize_tcanvas(klass):
    # Parameters:
    # klass: class to be pythonized

    klass._Update = klass.Update
    klass._Draw = klass.Draw
    klass.Update = _TCanvas_Update
    klass.Draw = _TCanvas_Draw
