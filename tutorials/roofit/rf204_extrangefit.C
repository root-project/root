/// \file
/// \ingroup tutorial_roofit
/// \notebook -nodraw
///
///  'ADDITION AND CONVOLUTION' RooFit tutorial macro #204
///
///  Extended maximum likelihood fit with alternate range definition
///  for observed number of events.
///  If multiple ranges are used, or only a part of the data is fitted,
///  it is advisable to use a RooAddPdf to extend the model. See tutorial
///  204a.
///
/// \macro_output
/// \macro_code
///
/// \date 07/2008
/// \author Wouter Verkerke

#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooChebychev.h"
#include "RooAddPdf.h"
#include "RooExtendPdf.h"
#include "RooFitResult.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "RooPlot.h"
using namespace RooFit;

void rf204_extrangefit()
{

   // S e t u p   c o m p o n e n t   p d f s
   // ---------------------------------------

   // Declare observable x
   RooRealVar x("x", "x", 0, 10);

   // Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their parameters
   RooRealVar mean("mean", "mean of gaussians", 5);
   RooRealVar sigma1("sigma1", "width of gaussians", 0.5);
   RooRealVar sigma2("sigma2", "width of gaussians", 1);

   RooGaussian sig1("sig1", "Signal component 1", x, mean, sigma1);
   RooGaussian sig2("sig2", "Signal component 2", x, mean, sigma2);

   // Build Chebychev polynomial p.d.f.
   RooRealVar a0("a0", "a0", 0.5, 0., 1.);
   RooRealVar a1("a1", "a1", 0.2, 0., 1.);
   RooChebychev bkg("bkg", "Background", x, RooArgSet(a0, a1));

   // Sum the signal components into a composite signal p.d.f.
   RooRealVar sig1frac("sig1frac", "fraction of component 1 in signal", 0.8, 0., 1.);
   RooAddPdf sig("sig", "Signal", RooArgList(sig1, sig2), sig1frac);

   // C o n s t r u c t   e x t e n d e d   c o m p s   wi t h   r a n g e   s p e c
   // ------------------------------------------------------------------------------

   // Define signal range in which events counts are to be defined
   x.setRange("signalRange", 4, 6);

   // Associated nsig/nbkg as expected number of events with sig/bkg _in_the_range_ "signalRange"
   RooRealVar nsig("nsig", "number of signal events in signalRange", 500, 0., 10000) ;
   RooRealVar nbkg("nbkg", "number of background events in signalRange", 500, 0, 10000) ;

   // Use AddPdf to extend the model:
   RooAddPdf  model("model","(g1+g2)+a", RooArgList(bkg,sig), RooArgList(nbkg,nsig)) ;

   // Clone these models here because the interpretation of normalisation coefficients changes
   // when different ranges are used:
   RooAddPdf model2(model);
   RooAddPdf model3(model);


   // S a m p l e   d a t a ,   f i t   m o d e l
   // -------------------------------------------

   // Generate 1000 events from model so that nsig,nbkg come out to numbers <<500 in fit
   RooDataSet *data = model.generate(x, 1000);


   auto canv = new TCanvas("Canvas", "Canvas", 1500, 600);
   canv->Divide(3,1);

   // Fit full range
   // -------------------------------------------

   canv->cd(1);

   // Perform unbinned ML fit to data, full range
   RooFitResult* r = model.fitTo(*data,Save()) ;
   r->Print() ;

}
