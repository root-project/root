#####################################
#
# 'ORGANIZATION AND SIMULTANEOUS FITS' ROOT.RooFit tutorial macro #506
#
# ROOT.Tuning and customizing the ROOT.RooFit message logging facility
#
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf506_msgservice():
    # C r e a t e   p d f
    # --------------------

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

    data = model.generate(ROOT.RooArgSet(x), 10)

    # P r i n t   c o n f i g u r a t i o n   o f   m e s s a g e   s e r v i c e
    # ---------------------------------------------------------------------------

    # Print streams configuration
    ROOT.RooMsgService.instance().Print()
    print ""

    # A d d i n g   I n t e g r a t i o n   t o p i c   t o   e x i s t i n g   I N F O   s t r e a m
    # -----------------------------------------------------------------------------------------------

    # Print streams configuration
    ROOT.RooMsgService.instance().Print()
    print ""

    # Add Integration topic to existing INFO stream
    ROOT.RooMsgService.instance().getStream(1).addTopic(ROOT.RooFit.Integration)

    # Construct integral over gauss to demonstrate message stream
    igauss = gauss.createIntegral(ROOT.RooArgSet(x))
    igauss.Print()

    # Print streams configuration in verbose, also shows inactive streams
    print ""
    ROOT.RooMsgService.instance().Print()
    print ""

    # Remove stream
    ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Integration)

    # E x a m p l e s   o f   p d f   v a l u e   t r a c i n g   s t r e a m
    # -----------------------------------------------------------------------

    # Show DEBUG level message on function tracing, ROOT.RooGaussian only
    ROOT.RooMsgService.instance().addStream(
        ROOT.RooFit.DEBUG, ROOT.RooFit.Topic(ROOT.RooFit.Tracing), ROOT.RooFit.ClassName("RooGaussian"))

    # Perform a fit to generate some tracing messages
    model.fitTo(data, ROOT.RooFit.Verbose(ROOT.kTRUE))

    # Reset message service to default stream configuration
    ROOT.RooMsgService.instance().reset()

    # Show DEBUG level message on function tracing on all objects, output to
    # file
    ROOT.RooMsgService.instance().addStream(
        ROOT.RooFit.DEBUG, ROOT.RooFit.Topic(ROOT.RooFit.Tracing), ROOT.RooFit.OutputFile("rf506_debug.log"))

    # Perform a fit to generate some tracing messages
    model.fitTo(data, ROOT.RooFit.Verbose(ROOT.kTRUE))

    # Reset message service to default stream configuration
    ROOT.RooMsgService.instance().reset()

    # E x a m p l e   o f   a n o t h e r   d e b u g g i n g   s t r e a m
    # ---------------------------------------------------------------------

    # Show DEBUG level messages on client/server link state management
    ROOT.RooMsgService.instance().addStream(
        ROOT.RooFit.DEBUG, ROOT.RooFit.Topic(ROOT.RooFit.LinkStateMgmt))
    ROOT.RooMsgService.instance().Print("v")

    # Clone composite pdf g to trigger some link state management activity
    gprime = gauss.cloneTree()
    gprime.Print()

    # Reset message service to default stream configuration
    ROOT.RooMsgService.instance().reset()


if __name__ == "__main__":
    rf506_msgservice()
