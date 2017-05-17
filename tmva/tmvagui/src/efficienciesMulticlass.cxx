// @(#)Root/tmva $Id$
// Author: Kim Albertsson
/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVAGUI                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors :                                                                      *
 *      Kim Albertsson  <kim.albertsson@cern.ch> - LTU & CERN                     *
 *                                                                                *
 * Copyright (c) 2005-2017:                                                       *
 *      CERN, Switzerland                                                         *
 *      LTU, Sweden                                                               *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/efficienciesMulticlass.h"

#include "TFile.h"
#include "TGraph.h"
#include "TH2F.h"
#include "TIterator.h"
#include "TKey.h"

////////////////////////////////////////////////////////////////////////////////
///
/// Note: This file assumes a certain structure on the input file. The structure
/// is as follows:
///
/// - dataset (TDirectory)
/// - ... some variables, plots ...
/// - Method_XXX (TDirectory)
///   + XXX (TDirectory)
///      * ... some plots ...
///      * MVA_Method_XXX_Test_#classname#
///      * ... some plots ...
/// - Method_YYY (TDirectory)
///   + YYY (TDirectory)
///      * ... some plots ...
///      * MVA_Method_YYY_Test_#classname#
///      * ... some plots ...
/// - TestTree (TTree)
///   + ... data...
/// - TrainTree (TTree)
///   + ... data...
///
/// Keeping this in mind makes the main loop in plotEfficienciesMulticlass easier
/// to follow :)
///

////////////////////////////////////////////////////////////////////////////////
/// Wrapper for a canvas that also keeps track of color assignments for added
///  subgraphs.

class EfficiencyPlotWrapper {

public:
   TCanvas *fCanvas;
   TLegend *fLegend;

   TString fClassname;
   Int_t fColor;

   UInt_t fNumMethods;

   EfficiencyPlotWrapper(TString classname);
   Int_t addGraph(TGraph *graph);

   void addLegendEntry(TString methodTitle, TGraph *graph);

private:
   Float_t fx0L;
   Float_t fdxL;
   Float_t fy0H;
   Float_t fdyH;

   TCanvas *newEfficiencyCanvas(TString className);
   TLegend *newEfficiencyLegend();
};

////////////////////////////////////////////////////////////////////////////////
/// Constructs a new canvas + auxiliary data for showing an efficiency plot.
///

EfficiencyPlotWrapper::EfficiencyPlotWrapper(TString classname)
{
   // Legend extents (init before calling newEfficiencyLegend...)
   fx0L = 0.107;
   fy0H = 0.899;
   fdxL = 0.457 - fx0L;
   fdyH = 0.22;
   fx0L = 0.15;
   fy0H = 1 - fy0H + fdyH + 0.07;

   fColor = 1;
   fNumMethods = 0;

   fClassname = classname;
   fCanvas = newEfficiencyCanvas(classname);
   fLegend = newEfficiencyLegend();
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a new graph to the plot. The added graph should contain a single ROC
/// curve.
///

Int_t EfficiencyPlotWrapper::addGraph(TGraph *graph)
{
   graph->SetLineWidth(3);
   graph->SetLineColor(fColor);
   fColor++;
   if (fColor == 5 || fColor == 10 || fColor == 11) {
      fColor++;
   }

   fCanvas->cd();
   graph->Draw("");
   fCanvas->Update();

   ++fNumMethods;

   return fColor;
}

////////////////////////////////////////////////////////////////////////////////
/// WARNING: Uses the current color, thus the correct call ordering is:
/// 	plotWrapper->addGraph(...);
/// 	plotWrapper->addLegendEntry(...);
///

void EfficiencyPlotWrapper::addLegendEntry(TString methodTitle, TGraph *graph)
{
   fLegend->AddEntry(graph, methodTitle, "l");

   Float_t dyH_local = fdyH * (Float_t(TMath::Min((UInt_t)10, fNumMethods) - 3.0) / 4.0);
   fLegend->SetY2(fy0H + dyH_local);

   fLegend->Paint();
   fCanvas->Update();
}

////////////////////////////////////////////////////////////////////////////////
/// Helper to create new Canvas

TCanvas *EfficiencyPlotWrapper::newEfficiencyCanvas(TString className)
{
   TString canvas_name = Form("%s_%s", className.Data(), "canvas");
   TString canvas_title = Form("ROC Curve %s", className.Data());
   TCanvas *c = new TCanvas(canvas_name, canvas_title, 200, 0, 650, 500);
   // global style settings
   c->SetGrid();
   c->SetTicks();

   // Frame
   TString xtit = "Signal Efficiency";
   TString ytit = "Background Rejection (1 - eff)";
   TString ftit = Form("Background Rejection vs Signal Efficiency %s", className.Data());
   Double_t x1 = 0.0;
   Double_t x2 = 1.0;
   Double_t y1 = 0.0;
   Double_t y2 = 1.0;

   TH2F *frame = new TH2F(Form("%s_%s", className.Data(), "frame"), ftit, 500, x1, x2, 500, y1, y2);
   frame->GetXaxis()->SetTitle(xtit);
   frame->GetYaxis()->SetTitle(ytit);
   TMVA::TMVAGlob::SetFrameStyle(frame, 1.0);
   frame->DrawClone();

   return c;
}

////////////////////////////////////////////////////////////////////////////////
/// Helper to create new legend.

TLegend *EfficiencyPlotWrapper::newEfficiencyLegend()
{
   TLegend *legend = new TLegend(fx0L, fy0H - fdyH, fx0L + fdxL, fy0H);
   // legend->SetTextSize( 0.05 );
   legend->SetHeader("MVA Method:");
   legend->SetMargin(0.4);
   legend->Draw("");

   return legend;
}

////////////////////////////////////////////////////////////////////////////////
/// Entry point. Called from the TMVAMulticlassGui Buttons
///
/// @param dataset        Dataset to operate on. Should be created by the TMVA Multiclass Factory.
/// @param filename_input Name of the input file procuded by a TMVA Multiclass Factory.
/// @param plotType       Specified what kind of ROC curve to draw. Currently only rejB vs. effS is supported.

void TMVA::efficienciesMulticlass(TString dataset, TString filename_input, EEfficiencyPlotType plotType,
                                  Bool_t useTMVAStyle)
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize(useTMVAStyle);

   // checks if filename_input is already open, and if not opens one
   TFile *file = TMVAGlob::OpenFile(filename_input);
   if (file == nullptr) {
      std::cout << "ERROR: filename \"" << filename_input << "\" is not found.";
      return;
   }

   plotEfficienciesMulticlass(plotType, file->GetDirectory(dataset.Data()));

   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Work horse function. Will operate on the currently open file (opened by
/// efficienciesMulticlass).
///
/// @param plotType See effcienciesMulticlass.
/// @param binDir   Directory in the file on which to operate.

void TMVA::plotEfficienciesMulticlass(EEfficiencyPlotType plotType, TDirectory *binDir)
{
   // The current multiclass version implements only type 2 - rejB vs effS
   if (plotType != EEfficiencyPlotType::kRejBvsEffS) {
      std::cout << "Error: For multiclass, only rejB vs effS is currently implemented.";
   }

   TString methodPrefix = "MVA_";
   TString graphNameRef = "rejBvsS";
   std::map<TString, EfficiencyPlotWrapper *> classCanvasMap;

   TList methods;
   UInt_t nm = TMVAGlob::GetListOfMethods(methods, binDir);
   if (nm == 0) {
      cout << "ups .. no methods found in to plot ROC curve for ... give up" << endl;
      return;
   }
   //   TIter next(file->GetListOfKeys());
   TIter next(&methods);

   // Loop over all method categories
   TKey *key;
   while ((key = (TKey *)next())) {
      TDirectory *mDir = (TDirectory *)key->ReadObj();
      TList titles;
      TMVAGlob::GetListOfTitles(mDir, titles);

      // Loop over each method within a category
      TIter nextTitle(&titles);
      TKey *titkey;
      TDirectory *titDir;
      while ((titkey = TMVAGlob::NextKey(nextTitle, "TDirectory"))) {
         titDir = (TDirectory *)titkey->ReadObj();
         TString methodTitle;
         TMVAGlob::GetMethodTitle(methodTitle, titDir);

         // Loop through all plots for the method
         TIter nextKey(titDir->GetListOfKeys());
         TKey *hkey2;
         while ((hkey2 = TMVAGlob::NextKey(nextKey, "TGraph"))) {

            TGraph *h = (TGraph *)hkey2->ReadObj();
            TString hname = h->GetName();
            if (hname.Contains(graphNameRef) && hname.BeginsWith(methodPrefix) && not hname.Contains("Train")) {

               // Extract classname from plot name
               UInt_t index = hname.Last('_');
               TString classname = hname(index + 1, hname.Length() - (index + 1));

               EfficiencyPlotWrapper *plotWrapper;
               // Creating the class map lazily, TMVAGlob::GetClassNames is
               // bugged and reports more classes than there are. This method
               // does not.
               try {
                  plotWrapper = classCanvasMap.at(classname);
               } catch (...) {
                  plotWrapper = new EfficiencyPlotWrapper(classname);
                  classCanvasMap.emplace(classname.Data(), plotWrapper);
               }

               plotWrapper->addGraph(h);
               plotWrapper->addLegendEntry(methodTitle, h);
            }
         }
      }
   }
}