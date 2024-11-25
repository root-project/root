/// \file
/// \ingroup tutorial_roofit_main
/// \notebook -js
/// Organisation and simultaneous fits: using simultaneous pdfs to describe simultaneous
/// fits to multiple datasets
///
/// \macro_image
/// \macro_code
/// \macro_output
///
/// \date July 2008
/// \author Wouter Verkerke

#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooChebychev.h"
#include "RooAddPdf.h"
#include "RooSimultaneous.h"
#include "RooCategory.h"
#include "TCanvas.h"
#include "TAxis.h"
#include "RooPlot.h"
using namespace RooFit;

void rf501_simultaneouspdf()
{
   // C r e a t e   m o d e l   f o r   p h y s i c s   s a m p l e
   // -------------------------------------------------------------

   // Create observables
   RooRealVar x("x", "x", -8, 8);

   // Construct signal pdf
   RooRealVar mean("mean", "mean", 0, -8, 8);
   RooRealVar sigma("sigma", "sigma", 0.3, 0.1, 10);
   RooGaussian gx("gx", "gx", x, mean, sigma);

   // Construct background pdf
   RooRealVar a0("a0", "a0", -0.1, -1, 1);
   RooRealVar a1("a1", "a1", 0.004, -1, 1);
   RooChebychev px("px", "px", x, RooArgSet(a0, a1));

   // Construct composite pdf
   RooRealVar f("f", "f", 0.2, 0., 1.);
   RooAddPdf model("model", "model", RooArgList(gx, px), f);

   // C r e a t e   m o d e l   f o r   c o n t r o l   s a m p l e
   // --------------------------------------------------------------

   // Construct signal pdf.
   // NOTE that sigma is shared with the signal sample model
   RooRealVar mean_ctl("mean_ctl", "mean_ctl", -3, -8, 8);
   RooGaussian gx_ctl("gx_ctl", "gx_ctl", x, mean_ctl, sigma);

   // Construct the background pdf
   RooRealVar a0_ctl("a0_ctl", "a0_ctl", -0.1, -1, 1);
   RooRealVar a1_ctl("a1_ctl", "a1_ctl", 0.5, -0.1, 1);
   RooChebychev px_ctl("px_ctl", "px_ctl", x, RooArgSet(a0_ctl, a1_ctl));

   // Construct the composite model
   RooRealVar f_ctl("f_ctl", "f_ctl", 0.5, 0., 1.);
   RooAddPdf model_ctl("model_ctl", "model_ctl", RooArgList(gx_ctl, px_ctl), f_ctl);

   // G e n e r a t e   e v e n t s   f o r   b o t h   s a m p l e s
   // ---------------------------------------------------------------

   // Generate 1000 events in x and y from model
   std::unique_ptr<RooDataSet> data{model.generate({x}, 1000)};
   std::unique_ptr<RooDataSet> data_ctl{model_ctl.generate({x}, 2000)};

   // C r e a t e   i n d e x   c a t e g o r y   a n d   j o i n   s a m p l e s
   // ---------------------------------------------------------------------------

   // Define category to distinguish physics and control samples events
   RooCategory sample("sample", "sample");
   sample.defineType("physics");
   sample.defineType("control");

   // Construct combined dataset in (x,sample)
   RooDataSet combData("combData", "combined data", x, Index(sample),
                       Import({{"physics", data.get()}, {"control", data_ctl.get()}}));

   // C o n s t r u c t   a   s i m u l t a n e o u s   p d f   i n   ( x , s a m p l e )
   // -----------------------------------------------------------------------------------

   // Construct a simultaneous pdf using category sample as index:
   // associate model with the physics state and model_ctl with the control state
   RooSimultaneous simPdf("simPdf", "simultaneous pdf", {{"physics", &model}, {"control", &model_ctl}}, sample);

   // P e r f o r m   a   s i m u l t a n e o u s   f i t
   // ---------------------------------------------------

   // Perform simultaneous fit of model to data and model_ctl to data_ctl
   std::unique_ptr<RooFitResult> fitResult{simPdf.fitTo(combData, PrintLevel(-1), Save(), PrintLevel(-1))};
   fitResult->Print();

   // P l o t   m o d e l   s l i c e s   o n   d a t a    s l i c e s
   // ----------------------------------------------------------------

   // Make a frame for the physics sample
   RooPlot *frame1 = x.frame(Title("Physics sample"));

   // Plot all data tagged as physics sample
   combData.plotOn(frame1, Cut("sample==sample::physics"));

   // Plot "physics" slice of simultaneous pdf.
   // NBL You _must_ project the sample index category with data using ProjWData
   // as a RooSimultaneous makes no prediction on the shape in the index category
   // and can thus not be integrated.
   // In other words: Since the PDF doesn't know the number of events in the different
   // category states, it doesn't know how much of each component it has to project out.
   // This information is read from the data.
   simPdf.plotOn(frame1, Slice(sample, "physics"), ProjWData(sample, combData));
   simPdf.plotOn(frame1, Slice(sample, "physics"), Components("px"), ProjWData(sample, combData), LineStyle(kDashed));

   // The same plot for the control sample slice. We do this with a different
   // approach this time, for illustration purposes. Here, we are slicing the
   // dataset and then use the data slice for the projection, because then the
   // RooFit::Slice() becomes unnecessary. This approach is more general,
   // because you can plot sums of slices by using logical or in the Cut()
   // command.
   RooPlot *frame2 = x.frame(Bins(30), Title("Control sample"));
   std::unique_ptr<RooAbsData> slicedData{combData.reduce(Cut("sample==sample::control"))};
   slicedData->plotOn(frame2);
   simPdf.plotOn(frame2, ProjWData(sample, *slicedData));
   simPdf.plotOn(frame2, Components("px_ctl"), ProjWData(sample, *slicedData), LineStyle(kDashed));

   // The same plot for all the phase space. Here, we can just use the original
   // combined dataset.
   RooPlot *frame3 = x.frame(Title("Both samples"));
   combData.plotOn(frame3);
   simPdf.plotOn(frame3, ProjWData(sample, combData));
   simPdf.plotOn(frame3, Components("px,px_ctl"), ProjWData(sample, combData),
                 LineStyle(kDashed));

   TCanvas *c = new TCanvas("rf501_simultaneouspdf", "rf403_simultaneouspdf", 1200, 400);
   c->Divide(3);
   auto draw = [&](int i, RooPlot & frame) {
      c->cd(i);
      gPad->SetLeftMargin(0.15);
      frame.GetYaxis()->SetTitleOffset(1.4);
      frame.Draw();
   };
   draw(1, *frame1);
   draw(2, *frame2);
   draw(3, *frame3);
}
