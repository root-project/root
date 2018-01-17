# /
#
# 'ORGANIZATION AND SIMULTANEOUS FITS' ROOT.RooFit tutorial macro #504
#
# Using ROOT.RooSimWSTool to construct a simultaneous p.d.f that is built
# of variations of an input p.d.f
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf504_simwstool():
    # C r e a t e   m a s t e r   p d f
    # ---------------------------------

    # Construct gauss(x,m,s)
    x = ROOT.RooRealVar("x", "x", -10, 10)
    m = ROOT.RooRealVar("m", "m", 0, -10, 10)
    s = ROOT.RooRealVar("s", "s", 1, -10, 10)
    gauss = ROOT.RooGaussian("g", "g", x, m, s)

    # Construct poly(x,p0)
    p0 = ROOT.RooRealVar("p0", "p0", 0.01, 0., 1.)
    poly = ROOT.RooPolynomial("p", "p", x, ROOT.RooArgList(p0))

    # model = f*gauss(x) + (1-f)*poly(x)
    f = ROOT.RooRealVar("f", "f", 0.5, 0., 1.)
    model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(
        gauss, poly), ROOT.RooArgList(f))

    # C r e a t e   c a t e g o r y   o b s e r v a b l e s   f o r   s p l i t t i n g
    # ----------------------------------------------------------------------------------

    # Define two categories that can be used for splitting
    c = ROOT.RooCategory("c", "c")
    c.defineType("run1")
    c.defineType("run2")

    d = ROOT.RooCategory("d", "d")
    d.defineType("foo")
    d.defineType("bar")

    # S e t u p   S i m W S ROOT.T o o l
    # -----------------------------

    # Import ingredients in a workspace
    w = ROOT.RooWorkspace("w", "w")
    getattr(w, 'import')(ROOT.RooArgSet(model, c, d))

    # Make Sim builder tool
    sct = ROOT.RooSimWSTool(w)

    # B u i l d   a   s i m u l t a n e o u s   m o d e l   w i t h   o n e   s p l i t
    # ---------------------------------------------------------------------------------

    # Construct a simultaneous p.d.f with the following form
    #
    # model_run1(x) = f*gauss_run1(x,m_run1,s) + (1-f)*poly
    # model_run2(x) = f*gauss_run2(x,m_run2,s) + (1-f)*poly
    # simpdf(x,c) = model_run1(x) if c=="run1"
    #             = model_run2(x) if c=="run2"
    #
    # Returned p.d.f is owned by the workspace
    model_sim = sct.build("model_sim", "model",
                          ROOT.RooFit.SplitParam("m", "c"))

    # Print tree structure of model
    model_sim.Print("t")

    # Adjust model_sim parameters in workspace
    w.var("m_run1").setVal(-3)
    w.var("m_run2").setVal(+3)

    # Print contents of workspace
    w.Print("v")

    # B u i l d   a   s i m u l t a n e o u s   m o d e l   w i t h   p r o d u c t   s p l i t
    # -----------------------------------------------------------------------------------------

    # Build another simultaneous p.d.f using a composite split in states c X d
    model_sim2 = sct.build("model_sim2", "model",
                           ROOT.RooFit.SplitParam("p0", "c,d"))

    # Print tree structure of self model
    model_sim2.Print("t")


if __name__ == "__main__":
    rf504_simwstool()
