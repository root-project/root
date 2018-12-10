/// \file
/// \ingroup tutorial_roofit
/// \notebook -nodraw
/// Addition and convolution: extended maximum likelihood fit with alternate range definition for observed number of
/// events.
///
/// \macro_output
/// \macro_code
/// \author 07/2008 - Wouter Verkerke

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
   
   RooPlot * frame = x.frame(Title("Full range fitted"));
   data->plotOn(frame);
   model.plotOn(frame, VisualizeError(*r));
   model.plotOn(frame);
   model.paramOn(frame);
   frame->Draw();
   
   
   // Fit in two regions
   // -------------------------------------------
   
   canv->cd(2);
   x.setRange("left",  0., 4.);
   x.setRange("right", 6., 10.);
   
   RooFitResult* r2 = model2.fitTo(*data,
      Range("left,right"),
      Save()) ;
   r2->Print();
   
   
   RooPlot * frame2 = x.frame(Title("Fit in left/right sideband"));
   data->plotOn(frame2);
   model2.plotOn(frame2, VisualizeError(*r2));
   model2.plotOn(frame2);
   model2.paramOn(frame2);
   frame2->Draw();
   
   
   // Fit in one region
   // -------------------------------------------
   
   canv->cd(3);
   x.setRange("leftToMiddle",  0., 5.);
   
   RooFitResult* r3 = model3.fitTo(*data,
      Range("leftToMiddle"),
      Save()) ;
   r3->Print();
   
   
   RooPlot * frame3 = x.frame(Title("Fit from left to middle"));
   data->plotOn(frame3);
   model3.plotOn(frame3, VisualizeError(*r3));
   model3.plotOn(frame3);
   model3.paramOn(frame3);
   frame3->Draw();
   
   canv->Draw();
}
