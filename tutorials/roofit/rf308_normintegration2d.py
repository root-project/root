# /
#
# 'MULTIDIMENSIONAL MODELS' ROOT.RooFit tutorial macro #308
#
# Examples on normalization of p.d.f.s,
# integration of p.d.fs, construction
# of cumulative distribution functions from p.d.f.s
# in two dimensions
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf308_normintegration2d():
    # S e t u p   m o d e l
    # ---------------------

    # Create observables x,y
    x = ROOT.RooRealVar("x", "x", -10, 10)
    y = ROOT.RooRealVar("y", "y", -10, 10)

    # Create p.d.f. gaussx(x,-2,3), gaussy(y,2,2)
    gx = ROOT.RooGaussian(
        "gx", "gx", x, ROOT.RooFit.RooConst(-2), ROOT.RooFit.RooConst(3))
    gy = ROOT.RooGaussian(
        "gy", "gy", y, ROOT.RooFit.RooConst(+2), ROOT.RooFit.RooConst(2))

    # gxy = gx(x)*gy(y)
    gxy = ROOT.RooProdPdf("gxy", "gxy", ROOT.RooArgList(gx, gy))

    # R e t r i e v e   r a w  &   n o r m a l i z e d   v a l u e s   o f   R o o F i t   p . d . f . s
    # --------------------------------------------------------------------------------------------------

    # Return 'raw' unnormalized value of gx
    print "gxy = ", gxy.getVal()

    # Return value of gxy normalized over x _and_ y in range [-10,10]
    nset_xy = ROOT.RooArgSet(x, y)
    print "gx_Norm[x,y] = ", gxy.getVal(nset_xy)

    # Create object representing integral over gx
    # which is used to calculate  gx_Norm[x,y] == gx / gx_Int[x,y]
    igxy = gxy.createIntegral(ROOT.RooArgSet(x, y))
    print "gx_Int[x,y] = ", igxy.getVal()

    # NB: it is also possible to do the following

    # Return value of gxy normalized over x in range [-10,10] (i.e. treating y
    # as parameter)
    nset_x = ROOT.RooArgSet(x)
    print "gx_Norm[x] = ", gxy.getVal(nset_x)

    # Return value of gxy normalized over y in range [-10,10] (i.e. treating x
    # as parameter)
    nset_y = ROOT.RooArgSet(y)
    print "gx_Norm[y] = ", gxy.getVal(nset_y)

    # I n t e g r a t e   n o r m a l i z e d   p d f   o v e r   s u b r a n g e
    # ----------------------------------------------------------------------------

    # Define a range named "signal" in x from -5,5
    x.setRange("signal", -5, 5)
    y.setRange("signal", -3, 3)

    # Create an integral of gxy_Norm[x,y] over x and y in range "signal"
    # ROOT.This is the fraction of of p.d.f. gxy_Norm[x,y] which is in the
    # range named "signal"
    igxy_sig = gxy.createIntegral(ROOT.RooArgSet(x, y), ROOT.RooFit.NormSet(
        ROOT.RooArgSet(x, y)), ROOT.RooFit.Range("signal"))
    print "gx_Int[x,y|signal]_Norm[x,y] = ", igxy_sig.getVal()

    # C o n s t r u c t   c u m u l a t i v e   d i s t r i b u t i o n   f u n c t i o n   f r o m   p d f
    # -----------------------------------------------------------------------------------------------------

    # Create the cumulative distribution function of gx
    # i.e. calculate Int[-10,x] gx(x') dx'
    gxy_cdf = gxy.createCdf(ROOT.RooArgSet(x, y))

    # Plot cdf of gx versus x
    hh_cdf = gxy_cdf.createHistogram("hh_cdf", x, ROOT.RooFit.Binning(
        40), ROOT.RooFit.YVar(y, ROOT.RooFit.Binning(40)))
    hh_cdf.SetLineColor(ROOT.kBlue)

    c = ROOT.TCanvas("rf308_normintegration2d",
                     "rf308_normintegration2d", 600, 600)
    ROOT.gPad.SetLeftMargin(0.15)
    hh_cdf.GetZaxis().SetTitleOffset(1.8)
    hh_cdf.Draw("surf")

    c.SaveAs("rf308_normintegration2d.png")


if __name__ == "__main__":
    rf308_normintegration2d()
