#####################################
#
# 'NUMERIC ALGORITHM ROOT.TUNING' ROOT.RooFit tutorial macro #902
#
# Configuration and customization of how MC sampling algorithms
# on specific p.d.f.s are executed
#
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf902_numgenconfig():

    # A d j u s t   g l o b a l   MC   s a m p l i n g   s t r a t e g y
    # ------------------------------------------------------------------

    # Example p.d.f. for use below
    x = ROOT.RooRealVar("x", "x", 0, 10)
    model = ROOT.RooChebychev("model", "model", x, ROOT.RooArgList(
        ROOT.RooFit.RooConst(0), ROOT.RooFit.RooConst(0.5), ROOT.RooFit.RooConst(-0.1)))

    # Change global strategy for 1D sampling problems without conditional observable
    # (1st kFALSE) and without discrete observable (2nd kFALSE) from ROOT.RooFoamGenerator,
    # ( an interface to the ROOT.TFoam MC generator with adaptive subdivisioning strategy ) to ROOT.RooAcceptReject,
    # a plain accept/reject sampling algorithm [ ROOT.RooFit default before
    # ROOT 5.23/04 ]
    ROOT.RooAbsPdf.defaultGeneratorConfig().method1D(
        ROOT.kFALSE, ROOT.kFALSE).setLabel("RooAcceptReject")

    # Generate 10Kevt using ROOT.RooAcceptReject
    data_ar = model.generate(ROOT.RooArgSet(
        x), 10000, ROOT.RooFit.Verbose(ROOT.kTRUE))
    data_ar.Print()

    # A d j u s t i n g   d e f a u l t   c o n f i g   f o r   a   s p e c i f i c   p d f
    # -------------------------------------------------------------------------------------

    # Another possibility: associate custom MC sampling configuration as default for object 'model'
    # ROOT.The kTRUE argument will install a clone of the default configuration as specialized configuration
    # for self model if none existed so far
    model.specialGeneratorConfig(ROOT.kTRUE).method1D(
        ROOT.kFALSE, ROOT.kFALSE).setLabel("RooFoamGenerator")

    # A d j u s t i n g   p a r a m e t e r s   o f   a   s p e c i f i c   t e c h n i q u e
    # ---------------------------------------------------------------------------------------

    # Adjust maximum number of steps of ROOT.RooIntegrator1D in the global
    # default configuration
    ROOT.RooAbsPdf.defaultGeneratorConfig().getConfigSection(
        "RooAcceptReject").setRealValue("nTrial1D", 2000)

    # Example of how to change the parameters of a numeric integrator
    # (Each config section is a ROOT.RooArgSet with ROOT.RooRealVars holding real-valued parameters
    #  and ROOT.RooCategories holding parameters with a finite set of options)
    model.specialGeneratorConfig().getConfigSection(
        "RooFoamGenerator").setRealValue("chatLevel", 1)

    # Generate 10Kevt using ROOT.RooFoamGenerator (FOAM verbosity increased
    # with above chatLevel adjustment for illustration purposes)
    data_foam = model.generate(ROOT.RooArgSet(x), 10000, ROOT.RooFit.Verbose())
    data_foam.Print()


if __name__ == "__main__":
    rf902_numgenconfig()
