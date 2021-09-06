#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooChebychev.h"
#include "RooAddPdf.h"
#include "RooExtendPdf.h"
#include "RooArgSet.h"
#include "RooLinkedList.h"
#include "RooDataSet.h"
#include "RooFitResult.h"


void rf204_extrangefit()
{
    RooRealVar x = RooRealVar("x", "x", 0, 10);
    RooRealVar mean = RooRealVar("mean", "mean of gaussians", 5);
    RooRealVar sigma1 = RooRealVar("sigma1", "width of gaussians", 0.5);
    RooRealVar sigma2 = RooRealVar("sigma2", "width of gaussians", 1);
    RooGaussian sig1 = RooGaussian("sig1", "Signal component 1", x, mean, sigma1);
    RooGaussian sig2 = RooGaussian("sig2", "Signal component 2", x, mean, sigma2);
    RooRealVar a0 = RooRealVar("a0", "a0", 0.5, 0.0, 1.0);
    RooRealVar a1 = RooRealVar("a1", "a1", -0.2, 0.0, 1.0);
    RooChebychev bkg = RooChebychev("bkg", "Background", x, {a0, a1});
    RooRealVar sig1frac = RooRealVar("sig1frac", "fraction of component 1 in signal", 0.8, 0.0, 1.0);
    RooAddPdf sig = RooAddPdf("sig", "Signal", {sig1, sig2}, {sig1frac});
    x.setRange("signalRange", 4, 6);
    RooRealVar nsig = RooRealVar("nsig", "number of signal events in signalRange", 500, 0.0, 10000);
    RooRealVar nbkg = RooRealVar("nbkg", "number of background events in signalRange", 500, 0, 10000);
    RooExtendPdf esig = RooExtendPdf("esig", "extended signal pdf", sig, nsig, "signalRange");
    RooExtendPdf ebkg = RooExtendPdf("ebkg", "extended background pdf", bkg, nbkg, "signalRange");
    RooAddPdf model = RooAddPdf("model", "(g1+g2)+a", {ebkg, esig});
    RooDataSet* data = model.generate(RooArgSet(x), 1000);
    //auto r = model.fitTo(*data,RooLinkedList());
    RooFitResult* r = model.fitTo(*data,{"Extended(true)"},{"Save(true)"});
    //some reason this isn't working...
    //r->Print();
}
