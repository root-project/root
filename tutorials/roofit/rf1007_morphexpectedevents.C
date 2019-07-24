/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Morphing function for three parameters 
///
/// \macro_image
/// \macro_output
/// \macro_code
/// \author 04/2016 - Carsten Burgard
#include "RooLagrangianMorphing.h"
#include "RooStringVar.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TFolder.h"
#include "TAxis.h"
using namespace RooFit;

void rf1007_morphexpectedevents()
{
   // ---------------------------------------------------------
   // E f f e c t i v e   L a g r a n g i a n   M o r p h i n g
   // =========================================================


   // C r e a t e   m o r p h i n g   f u n c t i o n
   // ------------------------------------------------

   // Define identifiers for infilename and observable
   std::string infilename = "input/ggfhzz4l_2d.root";
   std::string observable = "base/pth";
   // Define identifier for input sample foldernames,
   // require three samples to describe two parameter morphing funciton
   std::vector<std::string> samplelist = {"s1", "s2", "s3"};

   // Construct list of input samples
   RooArgList inputs;
   for(auto const& sample: samplelist)
   {
      RooStringVar* v = new RooStringVar(sample.c_str(), sample.c_str(), sample.c_str());
      inputs.add(*v);
   }

   // Construct two parameter morphing function for the transverse 
   // momentum of the Higgs in the process ggF Higgs
   RooHCggfZZMorphPdf morphfunc("ggFHZZ", "ggFHZZ", infilename.c_str(), observable.c_str(), inputs);

   // Set morphing function at parameter configuration of standard model
   std::string standardmodel = "s1";
   morphfunc.setParameters(standardmodel.c_str());

   // Make TGraphErrors with expected events and uncertainity for different
   // values of 'kAzz'
   TGraphErrors* g = new TGraphErrors();
   size_t i=0;
   for(auto val:{-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.}){
     morphfunc.setParameter("kAzz",val);
     g->SetPoint     (i,val,morphfunc.expectedEvents());
     g->SetPointError(i,0.,morphfunc.expectedUncertainty());
     i++;
   }

   // Draw the graph on a canvas
   TCanvas* c = new TCanvas("c","rf1007_morphexpectedevents",0,0,400,400);
   c->cd();
   gPad->SetLeftMargin(0.15);

   g->SetMarkerStyle(20);
   g->Draw("AEP");
   g->GetXaxis()->SetTitle("#kappa_{AZZ}");
   g->GetYaxis()->SetTitle("expected events");
   g->SetMinimum(9.0);
   g->SetMaximum(10.0);
}
