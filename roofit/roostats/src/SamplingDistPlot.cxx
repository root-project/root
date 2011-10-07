// @(#)root/roostats:$Id$
// Authors: Sven Kreiss    June 2010
// Authors: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//____________________________________________________________________
/*
SamplingDistPlot : 

This class provides simple and straightforward utilities to plot SamplingDistribution
objects.
*/

#include "RooStats/SamplingDistPlot.h"

#include "RooRealVar.h"
#include "TStyle.h"
#include "TLine.h"
#include "TFile.h"
#include "TVirtualPad.h"  // for gPad

#include <algorithm>
#include <iostream>


#ifndef ROO_MSG_SERVICE
#include "RooMsgService.h"
#endif

/// ClassImp for building the THtml documentation of the class 
ClassImp(RooStats::SamplingDistPlot);

using namespace RooStats;

//_______________________________________________________
SamplingDistPlot::SamplingDistPlot(Int_t nbins) :
   fHist(0),
   fLegend(NULL),
   fItems(),
   fOtherItems(),
   fRooPlot(NULL),
   fLogXaxis(kFALSE),
   fLogYaxis(kFALSE),
   fApplyStyle(kTRUE),
   fFillStyle(3004)
{
  // SamplingDistPlot default constructor with bin size
  fIterator = fItems.MakeIterator();
  fIsWeighted = kFALSE;
  fBins = nbins;
  fMarkerType = 20;
  fColor = 1;
}

/*
//_______________________________________________________
SamplingDistPlot::SamplingDistPlot(const char* name, const char* title, Int_t nbins, Double_t xmin, Double_t xmax) :
   fHist(0),
   fLegend(NULL),
   fItems(),
   fOtherItems(),
   fRooPlot(NULL),
   fLogXaxis(kFALSE),
   fLogYaxis(kFALSE),
   fApplyStyle(kTRUE),
   fFillStyle(3004)
{
  // SamplingDistPlot constructor
  fHist = new TH1F(name, title, nbins, xmin, xmax);
  fIterator = fItems.MakeIterator();
  fIsWeighted = kFALSE;
  fBins = nbins;
  fMarkerType = 20;
  fColor = 1;
}
*/

//_______________________________________________________
Double_t SamplingDistPlot::AddSamplingDistribution(const SamplingDistribution *samplingDist, Option_t *drawOptions) {
   // adds sampling distribution (and normalizes if "NORMALIZE" is given as an option)

   fSamplingDistr = samplingDist->GetSamplingDistribution();
   SetSampleWeights(samplingDist);

   TString options(drawOptions);
   options.ToUpper();

   Double_t xmin = *(std::min_element(fSamplingDistr.begin(), fSamplingDistr.end()));
   Double_t xmax = *(std::max_element(fSamplingDistr.begin(), fSamplingDistr.end()));
   assert(fBins > 1); 
   Double_t xlow = xmin - 0.5*(xmax-xmin)/(fBins-1);
   Double_t xup  = xmax + 0.5*(xmax-xmin)/(fBins-1);

   fHist = new TH1F(samplingDist->GetName(), samplingDist->GetTitle(), fBins, xlow, xup);

   TString varName = samplingDist->GetVarName();
   fHist->GetXaxis()->SetTitle(varName.Data());
   if(varName.Length() > 0) fVarName = samplingDist->GetVarName().Data();


   std::vector<Double_t>::iterator valuesIt = fSamplingDistr.begin();
   for (int w_idx = 0; valuesIt != fSamplingDistr.end(); ++valuesIt, ++w_idx) {
      if (fIsWeighted) fHist->Fill(*valuesIt, fSampleWeights[w_idx]);
      else fHist->Fill(*valuesIt);
   }

   // NORMALIZATION
   fHist->Sumw2();
   double weightSum = 1.0;
   if(options.Contains("NORMALIZE")) {
      weightSum = fHist->Integral("width");
      fHist->Scale(1./weightSum);

      options.ReplaceAll("NORMALIZE", "");
      options.Strip();
   }


   //some basic aesthetics
   fHist->SetMarkerStyle(fMarkerType);
   fHist->SetMarkerColor(fColor);
   fHist->SetLineColor(fColor);

   fMarkerType++;
   fColor++;

   fHist->SetStats(kFALSE);

   addObject(fHist, options.Data());

   TString title = samplingDist->GetTitle();
   if(fLegend  &&  title.Length() > 0) fLegend->AddEntry(fHist, title, "L");

   return 1./weightSum;
}

//_______________________________________________________
Double_t SamplingDistPlot::AddSamplingDistributionShaded(const SamplingDistribution *samplingDist, Double_t minShaded, Double_t maxShaded, Option_t *drawOptions) {
   Double_t scaleFactor = AddSamplingDistribution(samplingDist, drawOptions);

   TH1F *shaded = (TH1F*)fHist->Clone((string(samplingDist->GetName())+string("_shaded")).c_str());
   shaded->SetFillStyle(fFillStyle++);
   shaded->SetLineWidth(0);

   for (int i=0; i<shaded->GetNbinsX(); ++i) {
      if (shaded->GetBinCenter(i) < minShaded || shaded->GetBinCenter(i) > maxShaded){
         shaded->SetBinContent(i,0);
      }
   }

   TString options(drawOptions);
   options.ToUpper();
   if(options.Contains("NORMALIZE")) {
      options.ReplaceAll("NORMALIZE", "");
      options.Strip();
   }
   addObject(shaded, options.Data());

   return scaleFactor;
}

void SamplingDistPlot::AddLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2, const char* title) {
   TLine *line = new TLine(x1, y1, x2, y2);
   line->SetLineWidth(3);
   line->SetLineColor(kBlack);

   if(fLegend  &&  title) fLegend->AddEntry(line, title, "L");

   addOtherObject(line, ""); // no options
}


//_______________________________________________________
void SamplingDistPlot::SetSampleWeights(const SamplingDistribution* samplingDist)
{
  //Determine if the sampling distribution has weights and store them

  fIsWeighted = kFALSE;

  if(samplingDist->GetSampleWeights().size() != 0){
    fIsWeighted = kTRUE;
    fSampleWeights = samplingDist->GetSampleWeights();
  }  

  return;
}

void SamplingDistPlot::addObject(TObject *obj, Option_t *drawOptions) 
{
  // Add a generic object to this plot. The specified options will be
  // used to Draw() this object later. The caller transfers ownership
  // of the object with this call, and the object will be deleted
  // when its containing plot object is destroyed.

  if(0 == obj) {
    std::cerr << fName << "::addObject: called with a null pointer" << std::endl;
    return;
  }

  fItems.Add(obj,drawOptions);

  return;
}
void SamplingDistPlot::addOtherObject(TObject *obj, Option_t *drawOptions)
{
  // Add a generic object to this plot. The specified options will be
  // used to Draw() this object later. The caller transfers ownership
  // of the object with this call, and the object will be deleted
  // when its containing plot object is destroyed.

  if(0 == obj) {
    std::cerr << fName << "::addOtherObject: called with a null pointer" << std::endl;
    return;
  }

  fOtherItems.Add(obj,drawOptions);

  return;
}

//_____________________________________________________________________________
void SamplingDistPlot::Draw(Option_t * /*options */) {
   // Draw this plot and all of the elements it contains. The specified options
   // only apply to the drawing of our frame. The options specified in our add...()
   // methods will be used to draw each object we contain.

   ApplyDefaultStyle();

   Float_t theMin(0.), theMax(0.), theYMax(0.);
   GetAbsoluteInterval(theMin, theMax, theYMax);

   RooRealVar xaxis("xaxis", fVarName.Data(), theMin, theMax);

   //L.M. by drawing many times we create a memory leak ???
   if (fRooPlot) delete fRooPlot;


   fRooPlot = xaxis.frame();
   fRooPlot->SetTitle("");
   fRooPlot->SetMaximum(theYMax);

   fIterator->Reset();
   TH1F *obj = 0;
   while ((obj = (TH1F*) fIterator->Next())) {
      //obj->Draw(fIterator->GetOption());
      // add cloned objects to avoid mem leaks
      TH1 * cloneObj = (TH1*)obj->Clone();
      cloneObj->SetDirectory(0);
      fRooPlot->addTH1(cloneObj, fIterator->GetOption());
   }

   TIterator *otherIt = fOtherItems.MakeIterator();
   TObject *otherObj = NULL;
   while ((otherObj = otherIt->Next())) {
      TObject * cloneObj = otherObj->Clone();
      fRooPlot->addObject(cloneObj, otherIt->GetOption());
   }
   delete otherIt;


   if(fLegend) fRooPlot->addObject(fLegend);

   if(bool(gStyle->GetOptLogx()) != fLogXaxis) {
      if(!fApplyStyle) coutW(Plotting) << "gStyle will be changed to adjust SetOptLogx(...)";
      gStyle->SetOptLogx(fLogXaxis);
   }
   if(bool(gStyle->GetOptLogy()) != fLogYaxis) {
      if(!fApplyStyle) coutW(Plotting) << "gStyle will be changed to adjust SetOptLogy(...)";
      gStyle->SetOptLogy(fLogYaxis);
   }
   fRooPlot->Draw();

   // apply this since gStyle does not work for RooPlot
   if (gPad) { 
      gPad->SetLogx(fLogXaxis);
      gPad->SetLogy(fLogYaxis);
   }

   return;
}

void SamplingDistPlot::ApplyDefaultStyle(void) {
   if(fApplyStyle) {
      // use plain black on white colors
      Int_t icol = 0;
      gStyle->SetFrameBorderMode( icol );
      gStyle->SetCanvasBorderMode( icol );
      gStyle->SetPadBorderMode( icol );
      gStyle->SetPadColor( icol );
      gStyle->SetCanvasColor( icol );
      gStyle->SetStatColor( icol );
      gStyle->SetFrameFillStyle( 0 );

      // set the paper & margin sizes
      gStyle->SetPaperSize( 20, 26 );

      if(fLegend) {
         fLegend->SetFillColor(0);
         fLegend->SetBorderSize(1);
      }
   }
}

//_____________________________________________________________________________
void SamplingDistPlot::GetAbsoluteInterval(Float_t &theMin, Float_t &theMax, Float_t &theYMax) const
{
  Float_t tmpmin = 999.;
  Float_t tmpmax = -999.;
  Float_t tmpYmax = -999.;


  fIterator->Reset();
  TH1F *obj = 0;
  while((obj = (TH1F*)fIterator->Next())) {
    if(obj->GetXaxis()->GetXmin() < tmpmin) tmpmin = obj->GetXaxis()->GetXmin();
    if(obj->GetXaxis()->GetXmax() > tmpmax) tmpmax = obj->GetXaxis()->GetXmax();
    if(obj->GetMaximum() > tmpYmax) tmpYmax = obj->GetMaximum() + 0.1*obj->GetMaximum();
  }

  theMin = tmpmin;
  theMax = tmpmax;
  theYMax = tmpYmax;

  return;
}

//_____________________________________________________________________________
void SamplingDistPlot::SetLineColor(Color_t color, const SamplingDistribution *samplDist) {
   // Sets line color for given sampling distribution and
   // fill color for the associated shaded TH1F.

   if (samplDist == 0) {
      fHist->SetLineColor(color);
   } else {
      fIterator->Reset();
      TH1F *obj = 0;

      TString shadedName(samplDist->GetName());
      shadedName += "_shaded";

      while ((obj = (TH1F*) fIterator->Next())) {
         if (!strcmp(obj->GetName(), samplDist->GetName())) {
            obj->SetLineColor(color);
            //break;
         }
         if (!strcmp(obj->GetName(), shadedName.Data())) {
            obj->SetLineColor(color);
            obj->SetFillColor(color);
            //break;
         }
      }
   }

   return;
}

//_____________________________________________________________________________
void SamplingDistPlot::SetLineWidth(Width_t lwidth, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fHist->SetLineWidth(lwidth);
  }
  else{
    fIterator->Reset();
    TH1F *obj = 0;
    while((obj = (TH1F*)fIterator->Next())) {
      if(!strcmp(obj->GetName(),samplDist->GetName())){
	obj->SetLineWidth(lwidth);
	break;
      }
    }
  }

  return;
}

//_____________________________________________________________________________
void SamplingDistPlot::SetLineStyle(Style_t style, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fHist->SetLineStyle(style);
  }
  else{
    fIterator->Reset();
    TH1F *obj = 0;
    while((obj = (TH1F*)fIterator->Next())) {
      if(!strcmp(obj->GetName(),samplDist->GetName())){
	obj->SetLineStyle(style);
	break;
      }
    }
  }

  return;
}

//_____________________________________________________________________________
void SamplingDistPlot::SetMarkerStyle(Style_t style, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fHist->SetMarkerStyle(style);
  }
  else{
    fIterator->Reset();
    TH1F *obj = 0;
    while((obj = (TH1F*)fIterator->Next())) {
      if(!strcmp(obj->GetName(),samplDist->GetName())){
	obj->SetMarkerStyle(style);
	break;
      }
    }
  }

  return;
}

//_____________________________________________________________________________
void SamplingDistPlot::SetMarkerColor(Color_t color, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fHist->SetMarkerColor(color);
  }
  else{
    fIterator->Reset();
    TH1F *obj = 0;
    while((obj = (TH1F*)fIterator->Next())) {
      if(!strcmp(obj->GetName(),samplDist->GetName())){
	obj->SetMarkerColor(color);
	break;
      }
    }
  }

  return;
}

//_____________________________________________________________________________
void SamplingDistPlot::SetMarkerSize(Size_t size, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fHist->SetMarkerSize(size);
  }
  else{
    fIterator->Reset();
    TH1F *obj = 0;
    while((obj = (TH1F*)fIterator->Next())) {
      if(!strcmp(obj->GetName(),samplDist->GetName())){
	obj->SetMarkerSize(size);
	break;
      }
    }
  }

  return;
}

//_____________________________________________________________________________
TH1F* SamplingDistPlot::GetTH1F(const SamplingDistribution *samplDist)
{
  if(samplDist == NULL){
    return fHist;
  }else{
    fIterator->Reset();
    TH1F *obj = 0;
    while((obj = (TH1F*)fIterator->Next())) {
      if(!strcmp(obj->GetName(),samplDist->GetName())){
        return obj;
      }
    }
  }

  return NULL;
}


//_____________________________________________________________________________
void SamplingDistPlot::RebinDistribution(Int_t rebinFactor, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fHist->Rebin(rebinFactor);
  }
  else{
    fIterator->Reset();
    TH1F *obj = 0;
    while((obj = (TH1F*)fIterator->Next())) {
      if(!strcmp(obj->GetName(),samplDist->GetName())){
	obj->Rebin(rebinFactor);
	break;
      }
    }
  }

  return;
}


// TODO test
void SamplingDistPlot::DumpToFile(const char* RootFileName, Option_t *option, const char *ftitle, Int_t compress) {
   // All the objects are written to rootfile

   if(!fRooPlot) {
      cout << "Plot was not drawn yet. Dump can only be saved after it was drawn with Draw()." << endl;
      return;
   }

   TFile ofile(RootFileName, option, ftitle, compress);
   ofile.cd();
   fRooPlot->Write();
   ofile.Close();
}

