// @(#)root/roostats:$Id: LikelihoodIntervalPlot.h 26427 2009-05-20 15:45:36Z pellicci $

/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 * Authors:                                                              *
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke       *
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//____________________________________________________________________
/*
LikelihoodIntervalPlot : 

This class provides simple and straightforward utilities to plot a LikelihoodInterval
object.
*/

#include "RooStats/LikelihoodIntervalPlot.h"

#include <algorithm>
#include <iostream>

#include "TROOT.h"
#include "TMath.h"
#include "TLine.h"
#include "TObjArray.h"
#include "TList.h"
#include "TGraph.h"
#include "TPad.h"

#include "RooRealVar.h"
#include "RooPlot.h"
//#include "RooProfileLL.h"
#include "TF1.h"

/// ClassImp for building the THtml documentation of the class 
ClassImp(RooStats::LikelihoodIntervalPlot);

using namespace RooStats;

//_______________________________________________________
LikelihoodIntervalPlot::LikelihoodIntervalPlot()
{
  // LikelihoodIntervalPlot default constructor
  fInterval = 0;
  fNdimPlot = 0;
  fParamsPlot = 0;
  fColor = 0;
  fFillStyle = 4050; // half transparent
  fLineColor = 0;
  fMaximum = 2.;
  fNPoints = 40;
}

//_______________________________________________________
LikelihoodIntervalPlot::LikelihoodIntervalPlot(LikelihoodInterval* theInterval)
{
  // LikelihoodIntervalPlot constructor
  fInterval = theInterval;
  fParamsPlot = fInterval->GetParameters();
  fNdimPlot = fParamsPlot->getSize();
  fColor = kBlue;
  fLineColor = kGreen;
  fFillStyle = 4050; // half transparent
  fMaximum = 2.;
  fNPoints = 40;
}

//_______________________________________________________
LikelihoodIntervalPlot::~LikelihoodIntervalPlot()
{
  // LikelihoodIntervalPlot destructor
}

//_____________________________________________________________________________
void LikelihoodIntervalPlot::SetLikelihoodInterval(LikelihoodInterval* theInterval)
{
  fInterval = theInterval;
  fParamsPlot = fInterval->GetParameters();
  fNdimPlot = fParamsPlot->getSize();

  return;
}

//_____________________________________________________________________________
void LikelihoodIntervalPlot::SetPlotParameters(const RooArgSet *params) 
{
  fNdimPlot = params->getSize();
  fParamsPlot = (RooArgSet*) params->clone((std::string(params->GetName())+"_clone").c_str());

  return;
}

//_____________________________________________________________________________
void LikelihoodIntervalPlot::Draw(const Option_t *options) 
{

   if(fNdimPlot > 2){
      std::cout << "LikelihoodIntervalPlot::Draw(" << GetName() 
                << ") ERROR: contours for more than 2 dimensions not implemented!" << std::endl;
      return;
   }
   
   TIter it = fParamsPlot->createIterator();
   RooRealVar *myparam = (RooRealVar*)it.Next();
      
   RooAbsReal* newProfile = fInterval->GetLikelihoodRatio(); 

   TString opt = options; 
   opt.ToLower(); 
   bool useRooPlot = opt.Contains("rooplot");
   opt.ReplaceAll("rooplot","");

   RooPlot * frame = 0; 
   
   if(fNdimPlot == 1){

      

      const Double_t xcont_min = fInterval->LowerLimit(*myparam);
      const Double_t xcont_max = fInterval->UpperLimit(*myparam);

      RooRealVar* myarg = (RooRealVar *) newProfile->getVariables()->find(myparam->GetName());
      double x1 = myarg->getMin(); 
      double x2 = myarg->getMax(); 


      // use TF1 for drawing the function
      if (!useRooPlot) { 

         // set a first estimate of range including 2 times upper and lower limit
         double xmin = std::max( x1, 2*xcont_min - xcont_max); 
         double xmax = std::min( x2, 2*xcont_max - xcont_min); 
         
         TF1 * tmp = newProfile->asTF(*myarg); 
         std::cout << "setting range to " << xmin << " , " << xmax << std::endl;
         tmp->SetRange(xmin, xmax);      
         tmp->SetNpx(fNPoints);

         // clone the function to avoid later to sample it
         TF1 * f1 = (TF1*) tmp->Clone(); 
         delete tmp;
         
         f1->SetTitle("- log profile likelihood ratio");
         TString name = TString(GetName()) + TString("_PLL_") + TString(myarg->GetName());
         f1->SetName(name);
         
         // set range for displaying x values where function <=  fMaximum
         // if no reasanable value found mantain first estimate
         x1 = xmin; x2 = xmax;  
         if (fMaximum > 0) { 
            double x0 = f1->GetX(0, xmin, xmax);
            // check that minimum is between xmin and xmax
            if ( x0 > x1 && x0 < x2) { 
               x1 = f1->GetX(fMaximum, xmin, x0); 
               x2 = f1->GetX(fMaximum, x0, xmax); 
               f1->SetMaximum(fMaximum);
            //std::cout << "setting range to " << x1 << " , " << x2 << " x0 = " << x0 << std::endl;
            }
         }
         
         f1->SetRange(x1,x2);
         
         
         f1->SetLineColor(kBlue);
         f1->GetXaxis()->SetTitle(myarg->GetName());
         f1->GetYaxis()->SetTitle("- log #lambda");
         f1->Draw(opt);

      } 
      else { 
         // use a RooPlot for drawing the PL function
         frame = myarg->frame();
         frame->SetTitle(GetTitle());
         frame->GetYaxis()->SetTitle("- log #lambda");
         //    frame->GetYaxis()->SetTitle("- log profile likelihood ratio");
         
         newProfile->plotOn(frame); 
         
         frame->SetMaximum(fMaximum);
         frame->SetMinimum(0.);
      }

      
      myarg->setVal(xcont_max);
      const Double_t Yat_Xmax = newProfile->getVal();
         
      TLine *Yline_cutoff = new TLine(x1,Yat_Xmax,x2,Yat_Xmax);
      TLine *Yline_min = new TLine(xcont_min,0.,xcont_min,Yat_Xmax);
      TLine *Yline_max = new TLine(xcont_max,0.,xcont_max,Yat_Xmax);
      
      Yline_cutoff->SetLineColor(fLineColor);
      Yline_min->SetLineColor(fLineColor);
      Yline_max->SetLineColor(fLineColor);
      
      if (!useRooPlot) { 
         // need to draw the line 
         Yline_cutoff->Draw();
         Yline_min->Draw();
         Yline_max->Draw();
      } 
      else { 
         // add line in the RooPlot
         frame->addObject(Yline_min);
         frame->addObject(Yline_max);
         frame->addObject(Yline_cutoff);
         frame->Draw();
      }
      

      return;
   }
   else if(fNdimPlot == 2){

      RooRealVar *myparamY = (RooRealVar*)it.Next();

      TH2F* hist2D = (TH2F*)newProfile->createHistogram("_hist2D",*myparamY,RooFit::YVar(*myparam),RooFit::Binning(fNPoints),RooFit::Scaling(kFALSE));


      hist2D->SetTitle(GetTitle());
      hist2D->SetStats(kFALSE);

      Double_t cont_level = TMath::ChisquareQuantile(fInterval->ConfidenceLevel(),fNdimPlot); // level for -2log LR
      cont_level = cont_level/2; // since we are plotting -log LR
      hist2D->SetContour(1,&cont_level);

      hist2D->SetFillColor(fColor); 
      hist2D->SetFillStyle(fFillStyle); 
      hist2D->SetLineColor(fLineColor);

      TString tmpOpt(options);

      if(!tmpOpt.Contains("CONT")) tmpOpt.Append("CONT");
      if(!tmpOpt.Contains("LIST")) tmpOpt.Append("LIST"); // if you want the contour TGraphs

      hist2D->Draw(tmpOpt.Data());
      //    hist2D->Draw("cont2,list,same");

      gPad->Update();  // needed for get list of specials 

      // get TGraphs and add them
      //    gROOT->GetListOfSpecials()->Print();
      TObjArray *contours = (TObjArray*) gROOT->GetListOfSpecials()->FindObject("contours"); 
      if(contours){
         TList *list = (TList*)contours->At(0); 
         TGraph *gr1 = (TGraph*)list->First();
         gr1->SetLineColor(kBlack);
         gr1->SetLineStyle(kDashed);
         gr1->Draw("same");
      } else{
         std::cout << "no countours found in ListOfSpecials" << std::endl;
      }

      return;
   }


   return;
}
