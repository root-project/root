// @(#)root/hist:$Id$
// Author: Rene Brun   10/12/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <iostream>
#include "TROOT.h"
#include "TClassRef.h"
#include "TVirtualPad.h"
#include "TRatioPlot.h"
#include "TBrowser.h"
#include "TH1.h"
#include "TF1.h"
#include "TPad.h"
#include "TString.h"
#include "TGraphAsymmErrors.h"
#include "TGraphErrors.h"
#include "TGaxis.h"
#include "TCanvas.h"
#include "TFrame.h"
#include "TMath.h"
#include "TLine.h"
#include "TVirtualFitter.h"
#include "TFitResult.h"

#define _(x) std::cout << #x;
#define __(x) std::cout << "[" << std::string(__FILE__).substr(std::string(__FILE__).find_last_of("/\\") + 1) << ":" <<__LINE__ << "] " << x << std::endl ;
#define var_dump(v) __(#v << "=" << (v));

ClassImp(TRatioPlot)

////////////////////////////////////////////////////////////////////////////////

/** \class TRatioPlot 
    \ingroup Hist
Class for displaying ratios, differences and fit residuals.

TRatioPlot has two constructors, one which accepts two histograms, and is responsible
for setting up the calculation of ratios and differences. This calculation is in part
delegated to TEfficiency. A single option can be given as a parameter, that is 
used to determine which procedure is chosen. The remaining option string is then
passed through to the calculation, if applicable.

## Ratios and differences

Available options are:
| Option     | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| errprop    | uses the histogram `TH1::Divide` method, yields symmetric errors    |
| diff       | subtracts the histograms                                     |
| grid / nogrid | enable (default) or disable automatic drawing of the 0(1) dashed line |

Begin_Macro(source)
{
   gStyle->SetOptStat(0);   
   auto c1 = new TCanvas("c1", "A ratio example");
   auto h1 = new TH1D("h1", "h1", 50, 0, 10);
   auto h2 = new TH1D("h2", "h2", 50, 0, 10);
   auto f1 = new TF1("f1", "exp(- x/[0] )");
   f1->SetParameter(0, 3);
   h1->FillRandom("f1", 1900);
   h2->FillRandom("f1", 2000); 
   h1->Sumw2();
   h2->Scale(1.9 / 2.);
   auto rp = new TRatioPlot(h1, h2, "rp", "rp", "pois", "hist", "E", "AP", 1., 1.);
   rp->SetTicks(0, 1);
   rp->Draw();
   return c1;
}
End_Macro



A second constructor only accepts a single histogram, but expects it to have a fitted
function. The function is used to calculate the residual between the fit and the 
histogram.

| Option     | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| errasym    | Uses calculated asymmetric errors from `TH1::GetBinErrorUp`/`TH1::GetBinErrorLow`. Note that you need to set `TH1::SetBinErrorOption` first |
| errfunc    | Uses \f$ \sqrt{f(x)} \f$ as the error |

The asymmetric error case uses the upper or lower error depending on whether the function value
is above or below the histogram bin content.


Begin_Macro(source)
{
   gStyle->SetOptStat(0);   
   auto c1 = new TCanvas("c1", "fit residual simple"); 
   auto h1 = new TH1D("h1", "h1", 50, -5, 5);
   h1->FillRandom("gaus", 2000);
   h1->Fit("gaus");
   c1->Clear(); // Fit does not draw into correct pad
   auto rp1 = new TRatioPlot((TH1*)h1->Clone(), "rp1", "rp1", "", "", "AP");
   rp1->Draw();
   rp1->GetLowYaxis()->SetTitle("ratio");
   rp1->GetUpYaxis()->SetTitle("entries");
   c1->Update();
   return c1;
}   
End_Macro

## Access to internal parts
You can access the internal objects that are used to construct the plot via a series of 
methods. `TRatioPlot::GetUpperPad` and `TRatioPlot::GetLowerPad` can be used to draw additional
elements on top of the existing ones.
`TRatioPlot::GetLowerRefGraph` returns a reference to the lower pad's graph that
is responsible for the range, which enables you to modify the range.

*/


////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot default constructor

TRatioPlot::TRatioPlot()
   : TPad()
{
}

TRatioPlot::~TRatioPlot()
{

   gROOT->GetListOfCleanups()->Remove(this);

   if (fUpperPad != 0) delete fUpperPad;
   if (fLowerPad != 0) delete fLowerPad;
   if (fTopPad != 0) delete fTopPad;

   if (fRatioGraph != 0) delete fRatioGraph;
   if (fConfidenceInterval1 != 0) delete fConfidenceInterval1;
   if (fConfidenceInterval2 != 0) delete fConfidenceInterval2;

   for (unsigned int i=0;i<fGridlines.size();++i) {
      delete (fGridlines[i]);
   }

   if (fSharedXAxis != 0) delete fSharedXAxis;
   if (fUpperGXaxis != 0) delete fUpperGXaxis;
   if (fLowerGXaxis != 0) delete fLowerGXaxis;
   if (fUpperGYaxis != 0) delete fUpperGYaxis;
   if (fLowerGYaxis != 0) delete fLowerGYaxis;
   if (fUpperGXaxisMirror != 0) delete fUpperGYaxisMirror;
   if (fLowerGXaxisMirror != 0) delete fLowerGXaxisMirror;
   if (fUpperGYaxisMirror != 0) delete fUpperGYaxisMirror;
   if (fLowerGYaxisMirror != 0) delete fLowerGYaxisMirror;

   if (fUpYaxis != 0) delete fUpYaxis;
   if (fLowYaxis != 0) delete fLowYaxis;  

}

////////////////////////////////////////////////////////////////////////////////
/// Constructor for two histograms 
///
/// \param h1 First histogram 
/// \param h2 Second histogram 
/// \param name Name for the object
/// \param title Title for the object
/// \param displayOption Steers the error calculation, as well as ratio / difference
/// \param optH1 Drawing option for first histogram
/// \param optH2 Drawing option for second histogram
/// \param optGraph Drawing option the lower graph
/// \param c1 Scaling factor for h1
/// \param c2 Scaling factor for h2
// @TODO: Class should work with stacks as well
// @TODO !: Add option to have difference divided by error (significance)
// @TODO !!: Needs to have a destructor that actually cleans up
TRatioPlot::TRatioPlot(TH1* h1, TH1* h2, const char *name /*=0*/, const char *title /*=0*/, 
      Option_t *displayOption, Option_t *optH1, Option_t *optH2, Option_t *optGraph,
      Double_t c1, Double_t c2)
   : TPad(name, title, 0, 0, 1, 1),
     fH1(h1),
     fH2(h2),
     fGridlines() 
{
   gROOT->GetListOfCleanups()->Add(this);

   if (!fH1 || !fH2) {
      Warning("TRatioPlot", "Need two histograms.");
      return;
   }

   Bool_t h1IsTH1=fH1->IsA()->InheritsFrom(TH1::Class());
   Bool_t h2IsTH1=fH2->IsA()->InheritsFrom(TH1::Class());

   if (!h1IsTH1 && !h2IsTH1) {
      Warning("TRatioPlot", "Need two histograms deriving from TH2 or TH3.");
      return;
   }



   fParentPad = gPad;

   SetupPads();

   TString displayOptionString = TString(displayOption);

   if (displayOptionString.Contains("errprop")) {
      displayOptionString.ReplaceAll("errprop", "");
      fDisplayMode = TRatioPlot::CalculationMode::kDivideHist;
   } else if (displayOptionString.Contains("diffsig")) {
      displayOptionString.ReplaceAll("diffsig", "");
      fDisplayMode = TRatioPlot::CalculationMode::kDifferenceSign;   
      
      // determine which error style
      if (displayOptionString.Contains("errasym")) {
         fErrorMode = TRatioPlot::ErrorMode::kErrorAsymmetric; 
         displayOptionString.ReplaceAll("errasym", "");
      }

      if (displayOptionString.Contains("errfunc")) {
         fErrorMode = TRatioPlot::ErrorMode::kErrorFunc;
         displayOptionString.ReplaceAll("errfunc", "");
      } 
   } else if (displayOptionString.Contains("diff")) {
      displayOptionString.ReplaceAll("diff", "");
      fDisplayMode = TRatioPlot::CalculationMode::kDifference;   
   } else {
      fDisplayMode = TRatioPlot::CalculationMode::kDivideGraph;
   }

   if (displayOptionString.Contains("grid")) {
      displayOptionString.ReplaceAll("grid", "");
      fShowGridline = kTRUE;
   }
   if (displayOptionString.Contains("nogrid")) {
      displayOptionString.ReplaceAll("nogrid", "");
      fShowGridline = kFALSE;
   }
   
   fDisplayOption = displayOptionString;

   TString optH1String = TString(optH1);
   TString optH2String = TString(optH2);
   TString optGraphString = TString(optGraph);
   
   optH2String.ReplaceAll("same", "");
   optH2String.ReplaceAll("SAME", "");

   fOptH1 = optH1String;
   fOptH2 = optH2String;
   fOptGraph = optGraphString;

   fC1 = c1;
   fC2 = c2;

   // build ratio, everything is ready
   BuildLowerPlot();

   // taking x axis information from h1 by cloning it x axis
   fSharedXAxis = (TAxis*)(fH1->GetXaxis()->Clone());
   fUpYaxis = (TAxis*)(fH1->GetYaxis()->Clone());
   fLowYaxis = (TAxis*)(fRatioGraph->GetYaxis()->Clone());
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor for one histogram and a fit.
/// \param h1 The histogram 
/// \param name Name for the object
/// \param title Title for the object
/// \param displayOption Steers the error calculation
/// \param optH1 Drawing option for the histogram
/// \param optGraph Drawing option the lower graph
/// @TODO: add fitresult as optional argument
TRatioPlot::TRatioPlot(TH1* h1, const char *name, const char *title, Option_t *displayOption, Option_t *optH1,
         /*Option_t *fitOpt, */Option_t *optGraph, TFitResult *fitres) 
   : TPad(name, title, 0, 0, 1, 1),
     fH1(h1),
     fGridlines() 
{
   gROOT->GetListOfCleanups()->Add(this);

   if (!fH1) {
      Warning("TRatioPlot", "Need a histogram.");
      return;
   }

   Bool_t h1IsTH1=fH1->IsA()->InheritsFrom(TH1::Class());

   if (!h1IsTH1) {
      Warning("TRatioPlot", "Need a histogram deriving from TH2 or TH3.");
      return;
   }

   TList *h1Functions = fH1->GetListOfFunctions();

   if (h1Functions->GetSize() < 1) {
      Warning("TRatioPlot", "Histogram given needs to have a (fit) function associated with it");
      return;
   }

   fParentPad = gPad;
   
   fFitResult = fitres;

   fDisplayMode = TRatioPlot::CalculationMode::kFitResidual;
   
   TString displayOptionString = TString(displayOption);
   
   // determine which error style
   if (displayOptionString.Contains("errasym")) {
      fErrorMode = TRatioPlot::ErrorMode::kErrorAsymmetric; 
      displayOptionString.ReplaceAll("errasym", "");
   }

   if (displayOptionString.Contains("errfunc")) {
      fErrorMode = TRatioPlot::ErrorMode::kErrorFunc;
      displayOptionString.ReplaceAll("errfunc", "");
   }

   if (displayOptionString.Contains("grid")) {
      displayOptionString.ReplaceAll("grid", "");
      fShowGridline = kTRUE;
   }
   if (displayOptionString.Contains("nogrid")) {
      displayOptionString.ReplaceAll("nogrid", "");
      fShowGridline = kFALSE;
   }
   
   fDisplayOption = displayOptionString;

   BuildLowerPlot();

   fOptH1 = optH1;
   fOptGraph = optGraph;

   fSharedXAxis = (TAxis*)(fH1->GetXaxis()->Clone());
   fUpYaxis = (TAxis*)(fH1->GetYaxis()->Clone());
   fLowYaxis = (TAxis*)(fRatioGraph->GetYaxis()->Clone());
 
   //SyncAxesRanges();
   
   SetupPads();

}


void TRatioPlot::SetupPads() {

   // this method will delete all the pads before recreating them

   if (fUpperPad != 0) {
      delete fUpperPad;
      fUpperPad = 0;
   }

   if (fLowerPad != 0) {
      delete fLowerPad;
      fLowerPad = 0;
   }

   fUpperPad = new TPad(TString::Format("%s_%s", fName.Data(), "upper_pad"), "", 0., fSplitFraction, 1., 1.);
   fLowerPad = new TPad(TString::Format("%s_%s", fName.Data(), "lower_pad"), "", 0., 0., 1., fSplitFraction);

   SetLogx(fParentPad->GetLogx());
   SetLogy(fParentPad->GetLogy());

   fUpperPad->SetLogy(GetLogy());
   fUpperPad->SetLogx(GetLogx());
   fLowerPad->SetLogx(GetLogx());

   SetGridx(fParentPad->GetGridx());
   SetGridy(fParentPad->GetGridy());

   fUpperPad->SetGridx(GetGridx());
   fUpperPad->SetGridy(GetGridy());
   fLowerPad->SetGridx(GetGridx());
   fLowerPad->SetGridy(GetGridy());


   SetPadMargins();

   // connect to the pads signal
   fUpperPad->Connect("RangeAxisChanged()", "TRatioPlot", this, "RangeAxisChanged()");
   fLowerPad->Connect("RangeAxisChanged()", "TRatioPlot", this, "RangeAxisChanged()");

   fUpperPad->Connect("UnZoomed()", "TRatioPlot", this, "UnZoomed()");
   fLowerPad->Connect("UnZoomed()", "TRatioPlot", this, "UnZoomed()");

   fUpperPad->Connect("Resized()", "TRatioPlot", this, "SubPadResized()");
   fLowerPad->Connect("Resized()", "TRatioPlot", this, "SubPadResized()");


   if (fTopPad != 0) {
      delete fTopPad;
      fTopPad = 0;
   }

   Double_t margin = 0;
   fTopPad = new TPad(TString::Format("%s_%s", fName.Data(), "top_pad"), "", margin, margin, 1-margin, 1-margin);

   fTopPad->SetBit(kCannotPick);

}



////////////////////////////////////////////////////////////////////////////////
/// TRatioPlot copy constructor

TRatioPlot::TRatioPlot(const TRatioPlot &/*hrp*/)
{
   Warning("TRatioPlot", "Copy constructor not yet implemented");
    return;
}


////////////////////////////////////////////////////////////////////////////////
/// Browse.

void TRatioPlot::Browse(TBrowser *b)
{
   Draw(b ? b->GetDrawOption() : "");
   gPad->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the top margin of the upper pad.
/// \param margin The new margin
void TRatioPlot::SetUpTopMargin(Float_t margin)
{
   fUpTopMargin = margin;
   SetPadMargins();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the bottom margin of the upper pad.
/// \param margin The new margin
void TRatioPlot::SetUpBottomMargin(Float_t margin)
{
   fUpBottomMargin = margin;
   SetPadMargins();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the top margin of the lower pad.
/// \param margin The new margin
void TRatioPlot::SetLowTopMargin(Float_t margin)
{
   fLowTopMargin = margin;
   SetPadMargins();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the bottom margin of the lower pad.
/// \param margin The new margin
void TRatioPlot::SetLowBottomMargin(Float_t margin)
{
   fLowBottomMargin = margin;
   SetPadMargins();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the left margin of both pads.
/// \param margin The new margin
void TRatioPlot::SetLeftMargin(Float_t margin)
{
   fLeftMargin = margin;
   SetPadMargins();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the right margin of both pads.
/// \param margin The new margin
void TRatioPlot::SetRightMargin(Float_t margin)
{
   fRightMargin = margin;
   SetPadMargins();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the margin that separates the two pads. The margin is split according
/// to the relative sizes of the pads
/// \param margin The new margin
void TRatioPlot::SetSeparationMargin(Float_t margin)
{
   Float_t sf = fSplitFraction;
   fUpBottomMargin = margin/2./(1-sf);
   fLowTopMargin = margin/2./sf;
   SetPadMargins();
}



////////////////////////////////////////////////////////////////////////////////
/// Draw
void TRatioPlot::Draw(Option_t *option)
{

   //__("TRatioPlot::Draw called");

   TVirtualPad *padsav = gPad;
   fParentPad = gPad;

   TPad::Draw(option);

   // we are a TPad
   cd();

   fUpperPad->Draw();
   fLowerPad->Draw();

   fTopPad->SetFillStyle(0);
   fTopPad->Draw();


   fUpperPad->cd();

   // we need to hide the original axes of the hist 
   fH1->GetXaxis()->SetTickSize(0.);
   fH1->GetXaxis()->SetLabelSize(0.);
   fH1->GetXaxis()->SetTitleSize(0.);
   fH1->GetYaxis()->SetTickSize(0.);
   fH1->GetYaxis()->SetLabelSize(0.);
   fH1->GetYaxis()->SetTitleSize(0.);

   fH1->Draw(fOptH1);
   
   if (fH2 != 0) {
      fH2->Draw(fOptH2+"same");
   }

   fLowerPad->cd();
  
   fConfidenceInterval2->SetFillColor(fCi1Color);
   fConfidenceInterval1->SetFillColor(fCi2Color);
   
   if (fDisplayMode == TRatioPlot::CalculationMode::kFitResidual) {
      fConfidenceInterval2->Draw("A3");
      fConfidenceInterval1->Draw("3");
      fRatioGraph->Draw("LX+SAME");
   } else {
     
      TString opt = fOptGraph;
      fRatioGraph->Draw("A"+fOptGraph);
   
   }

   TAxis *refX = GetLowerRefXaxis();
   TAxis *refY = GetLowerRefYaxis();
   // hide visual axis of lower pad display
   refX->SetTickSize(0.);
   refX->SetLabelSize(0.);
   refX->SetTitleSize(0.);
   refY->SetTickSize(0.);
   refY->SetLabelSize(0.);
   refY->SetTitleSize(0.);


   // assign same axis ranges to lower pad as in upper pad
   // the visual axes will be created on paint
   SyncAxesRanges();

   CreateGridline(); 

   padsav->cd();

}

////////////////////////////////////////////////////////////////////////////////
/// Returns the reference graph for the lower pad, which means the graph that
/// is responsible for setting the coordinate system. It is the first graph
/// added to the primitive list of the lower pad.
/// This reference can be used to set the minimum and maximum of the lower pad. 
/// Note that `TRatioPlot::Draw` needs to have been called first, since the 
/// graphs are only created then.
///
/// Begin_Macro(source)
/// {
///    gStyle->SetOptStat(0);
///    auto c1 = new TCanvas("c1", "fit residual simple");
///    c1->SetLogy();
///    auto h1 = new TH1D("h1", "h1", 50, -5, 5);
///    h1->FillRandom("gaus", 2000);
///    h1->Fit("gaus");
///    h1->SetMinimum(0.001);
///    c1->Clear();
///    auto rp1 = new TRatioPlot(h1, "rp1", "rp1", "", "", "AP");
///    rp1->Draw();
///    rp1->GetLowerRefGraph()->SetMinimum(-2);
///    rp1->GetLowerRefGraph()->SetMaximum(2);
///    c1->Update();
///    return c1;
/// }
/// End_Macro
TGraph* TRatioPlot::GetLowerRefGraph() 
{
   if (fLowerPad == 0) {
      Error(__FUNCTION__, "Lower pad has not been defined");
      return 0;
   }

   TList *primlist = fLowerPad->GetListOfPrimitives();
   if (primlist->GetSize() == 0) {
      Error(__FUNCTION__, "Lower pad does not have primitives");
      return 0;
   }
   
   
   TObjLink *lnk = primlist->FirstLink();

   while (lnk) {
      TObject *obj = lnk->GetObject();
      
      if (obj->InheritsFrom(TGraph::Class())) {
         return (TGraph*)obj;
      }


      
      lnk = lnk->Next();
   }

   Error(__FUNCTION__, "Did not find graph in list");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Shortcut for
/// ~~~{.cpp}
/// rp->GetLowerRefGraph()->GetXaxis();
/// ~~~
TAxis* TRatioPlot::GetLowerRefXaxis()
{
   return GetLowerRefGraph()->GetXaxis(); 
}

////////////////////////////////////////////////////////////////////////////////
/// Shortcut for
/// ~~~{.cpp}
/// rp->GetLowerRefGraph()->GetYaxis();
/// ~~~
TAxis* TRatioPlot::GetLowerRefYaxis() 
{
   return GetLowerRefGraph()->GetYaxis(); 
}

// @TODO: Document CreateGridline
void TRatioPlot::CreateGridline()
{
   //__(__FUNCTION__);
   TVirtualPad *padsav = gPad;
   
   fLowerPad->cd();

   int curr = fGridlines.size();
   int dest = fGridlinePositions.size();

   if (curr > dest) {
      // we have too many
      //__("remove gridlines");
      for (int i=0;i<curr-dest;++i) {
         __(i);
         // kill the line
         delete fGridlines.at(i);
         // remove it from list
         fGridlines.erase(fGridlines.begin());
      }
   } else if (curr < dest) {
      // we don't have enough
      //__("add gridlines");
      
      for (int i=0;i<dest-curr;++i) {
         __(i);
         TLine *newline = new TLine(0, 0, 0, 0);
         newline->SetLineStyle(2);
         newline->Draw();
         fGridlines.push_back(newline);
      }
   } else {
      //__("no gridline mod");
      // nothing to do
   }
   
   Double_t first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());


   TLine *line;
   double y;
   for (int i=0;i<dest;++i) {
      line = fGridlines.at(i); 
      y = fGridlinePositions[i];

      //__("");
      //var_dump(i);
      //var_dump(first);
      //var_dump(last);
      //var_dump(y);

      line->SetX1(first);
      line->SetX2(last);
      line->SetY1(y);
      line->SetY2(y);

   }

   //if (fShowGridline) {
      //if (fGridline == 0) {
         //fGridline = new TLine(0, 0, 0, 0);
         //fGridline->SetLineStyle(2);
         //fGridline->Draw();
      //}

      //Double_t first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
      //Double_t last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());
      //Double_t y = 1;

      //if (
            //fDisplayMode == TRatioPlot::CalculationMode::kDifference 
            //|| fDisplayMode == TRatioPlot::CalculationMode::kFitResidual
            //|| fDisplayMode == TRatioPlot::CalculationMode::kDifferenceSign
      //) {
         //y = 0;
      //}

      //fGridline->SetX1(first);
      //fGridline->SetX2(last);
      //fGridline->SetY1(y);
      //fGridline->SetY2(y);
   //}


   padsav->cd();
}

////////////////////////////////////////////////////////////////////////////////
/// Does not really do anything right now, other than call super
void TRatioPlot::Paint(Option_t *opt) {
   TPad::Paint(opt);
}

////////////////////////////////////////////////////////////////////////////////
/// Creates the visual axes when painting.
void TRatioPlot::PaintModified()
{
      // sync y axes
   fH1->GetYaxis()->ImportAttributes(fUpYaxis);
   fRatioGraph->GetYaxis()->ImportAttributes(fLowYaxis);

   fH1->GetXaxis()->SetTickSize(0.);
   fH1->GetXaxis()->SetLabelSize(0.);
   fH1->GetXaxis()->SetTitleSize(0.);
   fH1->GetYaxis()->SetTickSize(0.);
   fH1->GetYaxis()->SetLabelSize(0.);
   fH1->GetYaxis()->SetTitleSize(0.);
   
   fRatioGraph->GetXaxis()->SetTickSize(0.);
   fRatioGraph->GetXaxis()->SetLabelSize(0.);
   fRatioGraph->GetXaxis()->SetTitleSize(0.);
   fRatioGraph->GetYaxis()->SetTickSize(0.);
   fRatioGraph->GetYaxis()->SetLabelSize(0.);
   fRatioGraph->GetYaxis()->SetTitleSize(0.);
   
   // create the visual axes
   CreateVisualAxes();
   CreateGridline();

   TPad::PaintModified();
   
   if (fIsUpdating) fIsUpdating = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Syncs the axes ranges from the shared ones to the actual ones.
void TRatioPlot::SyncAxesRanges()
{
   // get ranges from the shared axis clone
   Double_t first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());

   // set range on computed graph, have to set it twice becaus 
   // TGraph's axis looks strange otherwise
   fRatioGraph->GetXaxis()->SetLimits(first, last);
   fRatioGraph->GetXaxis()->SetRangeUser(first, last);
   
   TAxis *ref = GetLowerRefXaxis();

   ref->SetLimits(first, last);
   ref->SetRangeUser(first, last);

   fH1->GetXaxis()->SetRangeUser(first, last);

}

////////////////////////////////////////////////////////////////////////////////
/// Build the lower plot according to which constructor was called, and
/// which options were passed.
void TRatioPlot::BuildLowerPlot()
{
   // Clear and delete the graph if not exists
   if (fRatioGraph != 0) {
      fRatioGraph->IsA()->Destructor(fRatioGraph);
      fRatioGraph = 0;
   }

   if (fConfidenceInterval1 == 0) {
      fConfidenceInterval1 = new TGraphErrors();
   }
   
   if (fConfidenceInterval2 == 0) {
      fConfidenceInterval2 = new TGraphErrors();
   }



   static Double_t divideGridlines[] = {0.7, 1.0, 1.3};
   static Double_t diffGridlines[] = {2.0, 0.0, -2.0};
   static Double_t signGridlines[] = {1.0, 0.0, -1.0};

   // Determine the divide mode and create the lower graph accordingly
   // Pass divide options given in constructor
   if (fDisplayMode == TRatioPlot::CalculationMode::kDivideGraph) {
      // use TGraphAsymmErrors Divide method to create
     
      SetGridlines(divideGridlines, 3);

      TH1 *tmpH1 = (TH1*)fH1->Clone();
      TH1 *tmpH2 = (TH1*)fH2->Clone();

      tmpH1->Scale(fC1);
      tmpH2->Scale(fC2);

      TGraphAsymmErrors *ratioGraph = new TGraphAsymmErrors();
      ratioGraph->Divide(tmpH1, tmpH2, fDisplayOption.Data());
      fRatioGraph = ratioGraph;

      delete tmpH1;
      delete tmpH2;

   } else if (fDisplayMode == TRatioPlot::CalculationMode::kDifference) {
      SetGridlines(diffGridlines, 3);

      TH1 *tmpHist = (TH1*)fH1->Clone();

      tmpHist->Reset();

      tmpHist->Add(fH1, fH2, fC1, -1*fC2);
      fRatioGraph = new TGraphErrors(tmpHist);

      delete tmpHist;
   } else if (fDisplayMode == TRatioPlot::CalculationMode::kDifferenceSign) {

      SetGridlines(signGridlines, 3);
      
      fRatioGraph = new TGraphAsymmErrors();
      Int_t ipoint = 0;
      Double_t res;
      Double_t error;      

      Double_t x;
      Double_t val;
      Double_t val2;

      for (Int_t i=0; i<=fH1->GetNbinsX();++i) {
         val = fH1->GetBinContent(i);
         val2 = fH2->GetBinContent(i);
         x = fH1->GetBinCenter(i+1);
         
         if (fErrorMode == TRatioPlot::ErrorMode::kErrorAsymmetric) {

            Double_t errUp = fH1->GetBinErrorUp(i);
            Double_t errLow = fH1->GetBinErrorLow(i);

            if (val - val2 > 0) {
               // h1 > h2
               error = errLow;
            } else {
               // h1 < h2
               error = errUp;
            }


         } else if (fErrorMode == TRatioPlot::ErrorMode::kErrorSymmetric) {
            error = fH1->GetBinError(i);
         } else {
            Warning("TRatioPlot", "error mode is invalid");
            error = 0;
         }


         if (error != 0) {


            res = (val - val2) / error;

            ((TGraphAsymmErrors*)fRatioGraph)->SetPoint(ipoint, fH1->GetBinCenter(i), res);
            ((TGraphAsymmErrors*)fRatioGraph)->SetPointError(ipoint,  fH1->GetBinWidth(i)/2., fH1->GetBinWidth(i)/2., 0.5, 0.5);

            ++ipoint;

         }
      } 

   } else if (fDisplayMode == TRatioPlot::CalculationMode::kFitResidual) {
      
      SetGridlines(signGridlines, 3);
      
      TF1 *func = dynamic_cast<TF1*>(fH1->GetListOfFunctions()->At(0));

      fRatioGraph = new TGraphAsymmErrors();
      Int_t ipoint = 0;

      Double_t res;
      Double_t error;      

      std::vector<double> ci1;
      std::vector<double> ci2;

      
      Double_t x_arr[fH1->GetNbinsX()];
      Double_t ci_arr1[fH1->GetNbinsX()];
      Double_t ci_arr2[fH1->GetNbinsX()];
      for (Int_t i=0; i<fH1->GetNbinsX();++i) {
         x_arr[i] = fH1->GetBinCenter(i+1);
      }

      Double_t cl1 = fCl1;
      Double_t cl2 = fCl2;

      if (fFitResult != 0) {
         // use this to get conf int

         fFitResult->GetConfidenceIntervals(fH1->GetNbinsX(), 1, 1, x_arr, ci_arr1, cl1);
         for (Int_t i=1; i<=fH1->GetNbinsX();++i) {
            ci1.push_back(ci_arr1[i-1]);
         }
         
         fFitResult->GetConfidenceIntervals(fH1->GetNbinsX(), 1, 1, x_arr, ci_arr2, cl2);
         for (Int_t i=1; i<=fH1->GetNbinsX();++i) {
            ci2.push_back(ci_arr2[i-1]);
         }
      } else {
         (TVirtualFitter::GetFitter())->GetConfidenceIntervals(fH1->GetNbinsX(), 1, x_arr, ci_arr1, cl1);
         for (Int_t i=1; i<=fH1->GetNbinsX();++i) {
            ci1.push_back(ci_arr1[i-1]);
         }
         (TVirtualFitter::GetFitter())->GetConfidenceIntervals(fH1->GetNbinsX(), 1, x_arr, ci_arr2, cl2);
         for (Int_t i=1; i<=fH1->GetNbinsX();++i) {
            ci2.push_back(ci_arr2[i-1]);
         }

      }

      Double_t x;
      Double_t val;

      for (Int_t i=0; i<=fH1->GetNbinsX();++i) {
         val = fH1->GetBinContent(i);
         x = fH1->GetBinCenter(i+1);
          
         if (fErrorMode == TRatioPlot::ErrorMode::kErrorAsymmetric) {
            
            Double_t errUp = fH1->GetBinErrorUp(i);
            Double_t errLow = fH1->GetBinErrorLow(i);

            if (val - func->Eval(fH1->GetBinCenter(i)) > 0) {
               // h1 > fit
               error = errLow;
            } else {
               // h1 < fit
               error = errUp;
            }


         } else if (fErrorMode == TRatioPlot::ErrorMode::kErrorSymmetric) {
            error = fH1->GetBinError(i);
         } else if (fErrorMode == TRatioPlot::ErrorMode::kErrorFunc) {

            error = sqrt(func->Eval(x));

         } else {
            Warning("TRatioPlot", "error mode is invalid");
            error = 0;
         }


         if (error != 0) {
            

            res = (fH1->GetBinContent(i)- func->Eval(fH1->GetBinCenter(i) ) ) / error;
            //__("x="<< x << " y=" << res << " err=" << error);
           
            ((TGraphAsymmErrors*)fRatioGraph)->SetPoint(ipoint, fH1->GetBinCenter(i), res);
            ((TGraphAsymmErrors*)fRatioGraph)->SetPointError(ipoint,  fH1->GetBinWidth(i)/2., fH1->GetBinWidth(i)/2., 0.5, 0.5);

            fConfidenceInterval1->SetPoint(ipoint, x, 0);
            fConfidenceInterval1->SetPointError(ipoint, x, ci1[i] / error);
            fConfidenceInterval2->SetPoint(ipoint, x, 0);
            fConfidenceInterval2->SetPointError(ipoint, x, ci2[i] / error);

            ++ipoint;
            
         }

      }
 
   } else if (fDisplayMode == TRatioPlot::CalculationMode::kDivideHist){
      SetGridlines(divideGridlines, 3);
      
      // Use TH1's Divide method
      TH1 *tmpHist = (TH1*)fH1->Clone();
      tmpHist->Reset();
      tmpHist->Divide(fH1, fH2, fC1, fC2, fDisplayOption.Data());
      fRatioGraph = new TGraphErrors(tmpHist);
   
      delete tmpHist;
   }

   // need to set back to "" since recreation. we don't ever want
   // title on lower graph
   fRatioGraph->SetTitle("");
   fConfidenceInterval1->SetTitle("");
   fConfidenceInterval2->SetTitle("");
}


////////////////////////////////////////////////////////////////////////////////
/// (Re-)Creates the TGAxis objects that are used for consistent display of the 
/// axes.
void TRatioPlot::CreateVisualAxes()
{
   TVirtualPad *padsav = gPad;
   fTopPad->cd();

//   __(__PRETTY_FUNCTION__ << " called");

   // figure out where the axis has to go.
   // Implicit assumption is, that the top pad spans the full other pads
   Double_t upTM = fUpperPad->GetTopMargin();
   Double_t upBM = fUpperPad->GetBottomMargin();
   Double_t upLM = fUpperPad->GetLeftMargin();
   Double_t upRM = fUpperPad->GetRightMargin();

   Double_t lowTM = fLowerPad->GetTopMargin();
   Double_t lowBM = fLowerPad->GetBottomMargin();
   Double_t lowLM = fLowerPad->GetLeftMargin();
   Double_t lowRM = fLowerPad->GetRightMargin();

   Double_t first = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t last = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());

   Double_t upYFirst = fUpperPad->GetUymin();
   Double_t upYLast = fUpperPad->GetUymax();
   Double_t lowYFirst = fLowerPad->GetUymin();
   Double_t lowYLast = fLowerPad->GetUymax();

   Float_t sf = fSplitFraction;

   // check if gPad has the all sides axis set
   Bool_t mirroredAxes = fParentPad->GetFrameFillStyle() == 0 || GetFrameFillStyle() == 0; 
   Bool_t axistop = fTickx == 1 || mirroredAxes;
   Bool_t axisright = fTicky == 1 || mirroredAxes;

   Bool_t logx = fUpperPad->GetLogx() || fLowerPad->GetLogx();
   Bool_t uplogy = fUpperPad->GetLogy();
   Bool_t lowlogy = fLowerPad->GetLogy();

   if (uplogy) {
      
      upYFirst = TMath::Power(10, upYFirst);
      upYLast = TMath::Power(10, upYLast);
         
      if (upYFirst <= 0 || upYLast <= 0) {
         Error(__FUNCTION__, "Cannot set upper Y axis to log scale");
      }
   }

   if (lowlogy) {
      lowYFirst = TMath::Power(10, lowYFirst);
      lowYLast = TMath::Power(10, lowYLast);

      if (lowYFirst <= 0 || lowYLast <= 0) {
         Error(__FUNCTION__, "Cannot set lower Y axis to log scale");
      }

   }


   // this is different than in y, y already has pad coords converted, x not...
   if (logx) {
      if (first <= 0 || last <= 0) {
         Error(__FUNCTION__, "Cannot set X axis to log scale");
      }
   }
   

   // determine axes options to create log axes if needed
   TString xopt = "";
   if (logx) xopt.Append("G");
   TString upyopt = "";
   if (uplogy) upyopt.Append("G");
   TString lowyopt = "";
   if (lowlogy) lowyopt.Append("G");

   // only actually create them once, reuse otherwise b/c memory
   if (fUpperGXaxis == 0) {
      fUpperGXaxis = new TGaxis(0, 0, 1, 1, 0, 1, 510, "+U"+xopt);
      fUpperGXaxis->Draw();
   }

   if (fUpperGYaxis == 0) { 
      fUpperGYaxis = new TGaxis(0, 0, 1, 1, upYFirst, upYLast, 510, "S"+upyopt);
      fUpperGYaxis->Draw();   
   }   
   
   if (fLowerGXaxis == 0) { 
      fLowerGXaxis = new TGaxis(0, 0, 1, 1, first, last, 510, "+S"+xopt);
      fLowerGXaxis->Draw();
   }   
   
   if (fLowerGYaxis == 0) {
      fLowerGYaxis = new TGaxis(0, 0, 1, 1, lowYFirst, lowYLast, 510, "-S"+lowyopt);
      fLowerGYaxis->Draw();
   }

   // import infos from TAxes
   fUpperGXaxis->ImportAxisAttributes(fSharedXAxis);
   fUpperGYaxis->ImportAxisAttributes(fUpYaxis);
   fLowerGXaxis->ImportAxisAttributes(fSharedXAxis);
   fLowerGYaxis->ImportAxisAttributes(fLowYaxis);
   
   // (re)set all the axes properties to what we want them
   fUpperGXaxis->SetTitle("");

   fUpperGXaxis->SetX1(upLM);
   fUpperGXaxis->SetX2(1-upRM);
   fUpperGXaxis->SetY1(upBM*(1-sf)+sf);
   fUpperGXaxis->SetY2(upBM*(1-sf)+sf);
   fUpperGXaxis->SetWmin(first);
   fUpperGXaxis->SetWmax(last);

   fUpperGYaxis->SetX1(upLM);
   fUpperGYaxis->SetX2(upLM);
   fUpperGYaxis->SetY1(upBM*(1-sf)+sf);
   fUpperGYaxis->SetY2( (1-upTM)*(1-sf)+sf );
   fUpperGYaxis->SetWmin(upYFirst);
   fUpperGYaxis->SetWmax(upYLast);

   fLowerGXaxis->SetX1(lowLM);
   fLowerGXaxis->SetX2(1-lowRM);
   fLowerGXaxis->SetY1(lowBM*sf);
   fLowerGXaxis->SetY2(lowBM*sf);
   fLowerGXaxis->SetWmin(first);
   fLowerGXaxis->SetWmax(last);

   fLowerGYaxis->SetX1(lowLM);
   fLowerGYaxis->SetX2(lowLM);
   fLowerGYaxis->SetY1(lowBM*sf);
   fLowerGYaxis->SetY2((1-lowTM)*sf);
   fLowerGYaxis->SetWmin(lowYFirst);
   fLowerGYaxis->SetWmax(lowYLast);

   fUpperGXaxis->SetNdivisions(fSharedXAxis->GetNdivisions());
   fUpperGYaxis->SetNdivisions(fUpYaxis->GetNdivisions());
   fLowerGXaxis->SetNdivisions(fSharedXAxis->GetNdivisions());
   fLowerGYaxis->SetNdivisions(fLowYaxis->GetNdivisions());
   
   fUpperGXaxis->SetOption("+U"+xopt);
   fUpperGYaxis->SetOption("S"+upyopt);
   fLowerGXaxis->SetOption("+S"+xopt);
   fLowerGYaxis->SetOption("-S"+lowyopt);
 

   // normalize the tick sizes. y axis ticks should be consistent
   // even if their length is different
   Double_t ratio = ( (upBM-(1-upTM))*(1-sf) ) / ( (lowBM-(1-lowTM))*sf ) ;
   fUpperGXaxis->SetLabelSize(0.);
   Double_t ticksize = fUpperGYaxis->GetTickSize()*ratio;  
   fLowerGYaxis->SetTickSize(ticksize);

   
   // Create the axes on the other sides of the graphs 
   // This is steered by an option on the containing pad or self
   if (axistop || axisright) {
      
      // only actually create them once, reuse otherwise b/c memory
      if (fUpperGXaxisMirror == 0) {
         fUpperGXaxisMirror = (TGaxis*)fUpperGXaxis->Clone(); 
         if (axistop) fUpperGXaxisMirror->Draw(); 
      } 

      if (fLowerGXaxisMirror == 0) { 
         fLowerGXaxisMirror = (TGaxis*)fLowerGXaxis->Clone(); 
         if (axistop) fLowerGXaxisMirror->Draw();
      }

      if (fUpperGYaxisMirror == 0) {
         fUpperGYaxisMirror = (TGaxis*)fUpperGYaxis->Clone(); 
         if (axisright) fUpperGYaxisMirror->Draw();
      }
      
      if (fLowerGYaxisMirror == 0) {
         fLowerGYaxisMirror = (TGaxis*)fLowerGYaxis->Clone();
         if (axisright) fLowerGYaxisMirror->Draw(); 
      }

      // import attributes from shared axes
      fUpperGXaxisMirror->ImportAxisAttributes(fSharedXAxis);
      fUpperGYaxisMirror->ImportAxisAttributes(fUpYaxis);
      fLowerGXaxisMirror->ImportAxisAttributes(fSharedXAxis);
      fLowerGYaxisMirror->ImportAxisAttributes(fLowYaxis);

      // remove titles
      fUpperGXaxisMirror->SetTitle("");
      fUpperGYaxisMirror->SetTitle("");
      fLowerGXaxisMirror->SetTitle("");
      fLowerGYaxisMirror->SetTitle("");


      // move them about and set required positions
      fUpperGXaxisMirror->SetX1(upLM);
      fUpperGXaxisMirror->SetX2(1-upRM);
      fUpperGXaxisMirror->SetY1((1-upTM)*(1-sf)+sf);
      fUpperGXaxisMirror->SetY2((1-upTM)*(1-sf)+sf);
      fUpperGXaxisMirror->SetWmin(first);
      fUpperGXaxisMirror->SetWmax(last);
      
      fUpperGYaxisMirror->SetX1(1-upRM); 
      fUpperGYaxisMirror->SetX2(1-upRM); 
      fUpperGYaxisMirror->SetY1(upBM*(1-sf)+sf);
      fUpperGYaxisMirror->SetY2( (1-upTM)*(1-sf)+sf );
      fUpperGYaxisMirror->SetWmin(upYFirst);
      fUpperGYaxisMirror->SetWmax(upYLast);

      fLowerGXaxisMirror->SetX1(lowLM);
      fLowerGXaxisMirror->SetX2(1-lowRM);
      fLowerGXaxisMirror->SetY1((1-lowTM)*sf);
      fLowerGXaxisMirror->SetY2((1-lowTM)*sf); 
      fLowerGXaxisMirror->SetWmin(first);
      fLowerGXaxisMirror->SetWmax(last);
         
      fLowerGYaxisMirror->SetX1(1-lowRM); 
      fLowerGYaxisMirror->SetX2(1-lowRM); 
      fLowerGYaxisMirror->SetY1(lowBM*sf);
      fLowerGYaxisMirror->SetY2((1-lowTM)*sf);
      fLowerGYaxisMirror->SetWmin(lowYFirst);
      fLowerGYaxisMirror->SetWmax(lowYLast);

      // also needs normalized ticksize
      fLowerGYaxisMirror->SetTickSize(ticksize);

      fUpperGXaxisMirror->SetOption("-S"+xopt);
      fUpperGYaxisMirror->SetOption("+S"+upyopt);
      fLowerGXaxisMirror->SetOption("-S"+xopt);
      fLowerGYaxisMirror->SetOption("+S"+lowyopt);

      fUpperGXaxisMirror->SetNdivisions(fSharedXAxis->GetNdivisions());
      fUpperGYaxisMirror->SetNdivisions(fUpYaxis->GetNdivisions());
      fLowerGXaxisMirror->SetNdivisions(fSharedXAxis->GetNdivisions());
      fLowerGYaxisMirror->SetNdivisions(fLowYaxis->GetNdivisions());
      
      fUpperGXaxisMirror->SetLabelSize(0.);
      fLowerGXaxisMirror->SetLabelSize(0.);
      fUpperGYaxisMirror->SetLabelSize(0.);
      fLowerGYaxisMirror->SetLabelSize(0.);
   }


   padsav->cd();

}

////////////////////////////////////////////////////////////////////////////////
/// Sets the margins of all the pads to the value specified in class members.
/// This one is called whenever those are changed, e.g. in setters
void TRatioPlot::SetPadMargins()
{
   fUpperPad->SetTopMargin(fUpTopMargin);
   fUpperPad->SetBottomMargin(fUpBottomMargin);
   fUpperPad->SetLeftMargin(fLeftMargin);
   fUpperPad->SetRightMargin(fRightMargin);
   fLowerPad->SetTopMargin(fLowTopMargin);
   fLowerPad->SetBottomMargin(fLowBottomMargin);
   fLowerPad->SetLeftMargin(fLeftMargin);
   fLowerPad->SetRightMargin(fRightMargin);
}

////////////////////////////////////////////////////////////////////////////////
/// Figures out which pad margin has deviated from the stored ones,
/// to figure out what the new nominal is and set the other pad to it
/// subsequently.
Bool_t TRatioPlot::SyncPadMargins()
{

   Bool_t changed = kFALSE;

   if (fUpperPad->GetLeftMargin() != fLeftMargin) {
      fLeftMargin = fUpperPad->GetLeftMargin();
      changed = kTRUE;
   }
   else if (fLowerPad->GetLeftMargin() != fLeftMargin) {
      fLeftMargin = fLowerPad->GetLeftMargin();
      changed = kTRUE;
   }

   if (fUpperPad->GetRightMargin() != fRightMargin) {
      fRightMargin = fUpperPad->GetRightMargin();
      changed = kTRUE;
   }
   else if (fLowerPad->GetRightMargin() != fRightMargin) {
      fRightMargin = fLowerPad->GetRightMargin();
      changed = kTRUE;
   }

   // only reset margins, if any of the margins changed
   if (changed) {
      SetPadMargins();
   }


   Bool_t verticalChanged = kFALSE;

   if (fUpperPad->GetBottomMargin() != fUpBottomMargin) {

      verticalChanged = kTRUE;
      fUpBottomMargin = fUpperPad->GetBottomMargin();

   }
   
   if (fLowerPad->GetTopMargin() != fLowTopMargin) {

      verticalChanged = kTRUE;
      fLowTopMargin = fLowerPad->GetTopMargin();

   }
   
   if (fLowerPad->GetBottomMargin() != fLowBottomMargin) {

      fLowBottomMargin = fLowerPad->GetBottomMargin();

   }
   
   if (fUpperPad->GetTopMargin() != fUpTopMargin) {

      fUpTopMargin = fUpperPad->GetTopMargin();

   }

   // only reset margins, if any of the margins changed
   if (verticalChanged) {
      SetPadMargins();
   }

   return changed || verticalChanged;

}

////////////////////////////////////////////////////////////////////////////////
/// Slot that receives the RangeAxisChanged signal from any of the pads and
/// reacts correspondingly.
void TRatioPlot::RangeAxisChanged()
{
   // check if rp is already drawn.
   TList *siblings = fParentPad->GetListOfPrimitives();
   if (siblings->FindObject(this) == 0) {
      // not drawn yet
      return;
   }
 
   // Only run this concurrently once, in case it's called async
   if (fIsUpdating) {
      return;
   }

   fIsUpdating = kTRUE;
   

   // find out if logx has changed
   //var_dump(GetLogx());
   //var_dump(GetLogy());
   //var_dump(fUpperPad->GetLogx());
   //var_dump(fUpperPad->GetLogy());
   //var_dump(fLowerPad->GetLogx());
   //var_dump(fLowerPad->GetLogy());


   if (GetLogx()) {
      if (!fUpperPad->GetLogx() || !fLowerPad->GetLogx()) {
         SetLogx(kFALSE);
      }
   } else {
      if (fUpperPad->GetLogx() || fLowerPad->GetLogx()) {
         SetLogx(kTRUE);
      }
   }

   // set log to pad
   fUpperPad->SetLogx(GetLogx());
   fLowerPad->SetLogx(GetLogx());

   // copy logy from rp to upper pad
   //if (GetLogy() != 
   //fUpperPad->SetLogy(GetLogy());

   // get axis ranges for upper and lower 
   Double_t upFirst = fH1->GetXaxis()->GetBinLowEdge(fH1->GetXaxis()->GetFirst());
   Double_t upLast  = fH1->GetXaxis()->GetBinUpEdge(fH1->GetXaxis()->GetLast());

   //Double_t lowFirst = fRatioGraph->GetXaxis()->GetBinLowEdge(fRatioGraph->GetXaxis()->GetFirst());
   //Double_t lowLast  = fRatioGraph->GetXaxis()->GetBinUpEdge(fRatioGraph->GetXaxis()->GetLast());
   //Double_t lowFirst = fConfidenceIntervals->GetXaxis()->GetBinLowEdge(fConfidenceIntervals->GetXaxis()->GetFirst());
   //Double_t lowLast  = fConfidenceIntervals->GetXaxis()->GetBinUpEdge(fConfidenceIntervals->GetXaxis()->GetLast());

   TAxis *refX = GetLowerRefXaxis();
   Double_t lowFirst = refX->GetBinLowEdge(refX->GetFirst());
   Double_t lowLast = refX->GetBinUpEdge(refX->GetLast());

   Double_t globFirst = fSharedXAxis->GetBinLowEdge(fSharedXAxis->GetFirst());
   Double_t globLast = fSharedXAxis->GetBinUpEdge(fSharedXAxis->GetLast());

   Bool_t upChanged = kFALSE;
   Bool_t lowChanged = kFALSE;

   // determine which one has changed
   if (upFirst != globFirst || upLast != globLast) {
      fSharedXAxis->SetRangeUser(upFirst, upLast);
      upChanged = kTRUE;
   }
   else if (lowFirst != globFirst || lowLast != globLast) {
      fSharedXAxis->SetRangeUser(lowFirst, lowLast);
      lowChanged = kTRUE;
   }

   if (upChanged || lowChanged) {
      SyncAxesRanges();
      CreateVisualAxes();
      CreateGridline();   

      // @TODO: Fix updating, it's not working if zooming on lower axis
      fUpperPad->Modified();
      fLowerPad->Modified();
      fTopPad->Modified();
      Modified();
      fCanvas->Modified();
      fCanvas->Update();
   }

   // sync the margins in case the user has dragged one of them
   Bool_t marginsChanged = SyncPadMargins();

   if (marginsChanged) { 
      fUpperPad->Modified();
      fLowerPad->Modified();
      fTopPad->Modified();
      Modified();
      fCanvas->Modified();
      fCanvas->Update();
   }

   CreateVisualAxes();
   CreateGridline();   
   fIsUpdating = kFALSE;
}


////////////////////////////////////////////////////////////////////////////////
/// Slot for the UnZoomed signal that was introduced to TAxis. 
/// Unzooms both pads
void TRatioPlot::UnZoomed()
{
   // this is what resets the range
   fSharedXAxis->SetRange(0, 0);
   SyncAxesRanges();

   // Flushing
   fUpperPad->Modified();
   fLowerPad->Modified();
   fTopPad->Modified();
   Modified();
   fCanvas->Modified();
   fCanvas->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot that handles common resizing of upper and lower pad.
void TRatioPlot::SubPadResized() 
{

   if (fIsPadUpdating) {   
      return;
   }

   fIsPadUpdating = kTRUE;

   Float_t upylow = fUpperPad->GetYlowNDC();
   Float_t lowylow = fLowerPad->GetYlowNDC();
   Float_t lowh = fLowerPad->GetHNDC();
   Float_t lowyup = lowylow + lowh; 

   Bool_t changed = kFALSE;


   if (upylow != fSplitFraction) {
      // up changed
      SetSplitFraction(upylow);
      changed = kTRUE;
   }
   else if (lowyup != fSplitFraction) {
      // low changed
      SetSplitFraction(lowyup);
      changed = kTRUE;
   }
    
   if (changed) {
      CreateVisualAxes();
   }

   fIsPadUpdating = kFALSE;

}

Bool_t TRatioPlot::IsDrawn()
{
   TList *siblings = fParentPad->GetListOfPrimitives();
   return siblings->FindObject(this) != 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Set the fraction of the parent pad, at which the to sub pads should meet
void TRatioPlot::SetSplitFraction(Float_t sf) {
   fSplitFraction = sf;
   fUpperPad->SetPad(0., fSplitFraction, 1., 1.);
   fLowerPad->SetPad(0., 0., 1., fSplitFraction);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the confidence levels used to calculate the bands in the fit residual
/// case. Defaults to 1 and 2 sigma. You have to call TRatioPlot::BuildLowerPlot
/// to rebuild the bands.
void TRatioPlot::SetConfidenceLevels(Double_t c1, Double_t c2)
{
   fCl1 = c1;
   fCl2 = c2;
}
// @FIXME: SetLogy is not working interactively DONE
////////////////////////////////////////////////////////////////////////////////
/// Set logx for both of the pads
void TRatioPlot::SetLogx(Int_t value )
{
   TPad::SetLogx(value);
   fUpperPad->SetLogx(value); 
   fLowerPad->SetLogx(value); 
}

////////////////////////////////////////////////////////////////////////////////
/// Sets logy for the upper pad.
void TRatioPlot::SetLogy(Int_t value)
{
   TPad::SetLogy(value);
   fUpperPad->SetLogy(value);
}

void TRatioPlot::SetGridlines(std::vector<double> gridlines) 
{
   fGridlinePositions = gridlines;
}


void TRatioPlot::SetGridlines(Double_t *gridlines, Int_t numGridlines)
{
   fGridlinePositions.clear();
   //fGridlines.resize(numGridlines);
   
   for (Int_t i=0;i<numGridlines;++i) {
      fGridlinePositions.push_back(gridlines[i]);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \param ci1 Color of the 1 sigma band
/// \param ci2 Color of the 2 sigma band
/// Sets the color of the 1 and 2 sigma bands in the fit residual case.
/// Begin_Macro(source)
/// {
///    gStyle->SetOptStat(0);
///    auto c1 = new TCanvas("c1", "fit residual simple");
///    auto h1 = new TH1D("h1", "h1", 50, -5, 5);
///    h1->FillRandom("gaus", 2000);
///    h1->Fit("gaus");
///    c1->Clear();
///    auto rp1 = new TRatioPlot(h1, "rp1", "rp1", "nogrid");
///    rp1->SetConfidenceIntervalColors(kBlue, kRed);
///    rp1->Draw();
///    c1->Update();
///    return c1;
/// }
/// End_Macro
void TRatioPlot::SetConfidenceIntervalColors(Color_t ci1, Color_t ci2)
{
   fCi1Color = ci1; 
   fCi2Color = ci2; 
}

