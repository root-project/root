# /
#
# 'MULTIDIMENSIONAL MODELS' ROOT.RooFit tutorial macro #303
#
# Use of tailored p.d.f as conditional p.d.fs.s
#
# pdf = gauss(x,f(y),sx | y ) with f(y) = a0 + a1*y
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf303_conditional():
    # S e t u p   c o m p o s e d   m o d e l   g a u s s ( x , m ( y ) , s )
    # -----------------------------------------------------------------------

    # Create observables
    x = ROOT.RooRealVar("x", "x", -10, 10)
    y = ROOT.RooRealVar("y", "y", -10, 10)

    # Create function f(y) = a0 + a1*y
    a0 = ROOT.RooRealVar("a0", "a0", -0.5, -5, 5)
    a1 = ROOT.RooRealVar("a1", "a1", -0.5, -1, 1)
    fy = ROOT.RooPolyVar("fy", "fy", y, ROOT.RooArgList(a0, a1))

    # Creat gauss(x,f(y),s)
    sigma = ROOT.RooRealVar("sigma", "width of gaussian", 0.5, 0.1, 2.0)
    model = ROOT.RooGaussian(
        "model", "Gaussian with shifting mean", x, fy, sigma)

    # Obtain fake external experimental dataset with values for x and y
    expDataXY = makeFakeDataXY()

    # G e n e r a t e   d a t a   f r o m   c o n d i t i o n a l   p . d . f   m o d e l ( x | y )
    # ---------------------------------------------------------------------------------------------

    # Make subset of experimental data with only y values
    expDataY = expDataXY.reduce(ROOT.RooArgSet(y))

    # Generate 10000 events in x obtained from _conditional_ model(x|y) with y
    # values taken from experimental data
    data = model.generate(ROOT.RooArgSet(x), ROOT.RooFit.ProtoData(expDataY))
    data.Print()

    # F i t   c o n d i t i o n a l   p . d . f   m o d e l ( x | y )   t o   d a t a
    # ---------------------------------------------------------------------------------------------

    model.fitTo(expDataXY, ROOT.RooFit.ConditionalObservables(ROOT.RooArgSet(y)))

    # P r o j e c t   c o n d i t i o n a l   p . d . f   o n   x   a n d   y   d i m e n s i o n s
    # ---------------------------------------------------------------------------------------------

    # Plot x distribution of data and projection of model x = 1/Ndata
    # sum(data(y_i)) model(x;y_i)
    xframe = x.frame()
    expDataXY.plotOn(xframe)
    model.plotOn(xframe, ROOT.RooFit.ProjWData(expDataY))

    # Speed up (and approximate) projection by using binned clone of data for
    # projection
    binnedDataY = expDataY.binnedClone()
    model.plotOn(xframe, ROOT.RooFit.ProjWData(binnedDataY), ROOT.RooFit.LineColor(
        ROOT.kCyan), ROOT.RooFit.LineStyle(ROOT.kDotted))

    # Show effect of projection with too coarse binning
    (expDataY.get().find("y")).setBins(5)
    binnedDataY2 = expDataY.binnedClone()
    model.plotOn(xframe, ROOT.RooFit.ProjWData(
        binnedDataY2), ROOT.RooFit.LineColor(ROOT.kRed))

    # Make canvas and draw ROOT.RooPlots
    c = ROOT.TCanvas("rf303_conditional", "rf303_conditional", 600, 460)
    ROOT.gPad.SetLeftMargin(0.15)
    xframe.GetYaxis().SetTitleOffset(1.2)
    xframe.Draw()

    c.SaveAs("rf303_conditional.png")


def makeFakeDataXY():
    x = ROOT.RooRealVar("x", "x", -10, 10)
    y = ROOT.RooRealVar("y", "y", -10, 10)
    coord = ROOT.RooArgSet(x, y)

    d = ROOT.RooDataSet("d", "d", ROOT.RooArgSet(x, y))

    for i in range(10000):
        tmpy = ROOT.gRandom.Gaus(0, 10)
        tmpx = ROOT.gRandom.Gaus(0.5 * tmpy, 1)
        if (abs(tmpy) < 10) and (abs(tmpx) < 10):
            x = tmpx
            y = tmpy
            d.add(coord)

    return d


if __name__ == "__main__":
    rf303_conditional()
