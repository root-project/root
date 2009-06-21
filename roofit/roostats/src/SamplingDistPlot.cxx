// @(#)root/roostats:$Id: SamplingDistPlot.h 26427 2009-05-20 15:45:36Z pellicci $

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
SamplingDistPlot : 

This class provides simple and straightforward utilities to plot SamplingDistribution
objects.
*/

#include "RooStats/SamplingDistPlot.h"

#include "RooRealVar.h"
#include "RooPlot.h"

#include <algorithm>
#include <iostream>

/// ClassImp for building the THtml documentation of the class 
ClassImp(RooStats::SamplingDistPlot);

using namespace RooStats;

//_______________________________________________________
SamplingDistPlot::SamplingDistPlot() :
 fhist(0) ,fItems()
{
  // SamplingDistPlot default constructor
  fIterator = fItems.MakeIterator();
  fbins = 100;
  fMarkerType = 20;
  fColor = 1;
}

//_______________________________________________________
SamplingDistPlot::SamplingDistPlot(const Int_t nbins) :
 fhist(0) ,fItems()
{
  // SamplingDistPlot default constructor with bin size
  fIterator = fItems.MakeIterator();
  fbins = nbins;
  fMarkerType = 20;
  fColor = 1;
}


//_______________________________________________________
SamplingDistPlot::SamplingDistPlot(const char* name, const char* title, Int_t nbins, Double_t xmin, Double_t xmax) :
 fhist(0) ,fItems()
{
  // SamplingDistPlot constructor
  fhist = new TH1F(name, title, nbins, xmin, xmax);
  fbins = nbins;
  fMarkerType = 20;
  fColor = 1;
}

//_______________________________________________________
SamplingDistPlot::~SamplingDistPlot()
{
  // SamplingDistPlot destructor

  fSamplingDistr.clear();
  fSampleWeights.clear();

  fItems.Clear();
}

//_______________________________________________________
void SamplingDistPlot::AddSamplingDistribution(const SamplingDistribution *samplingDist, Option_t *drawOptions)
{
  fSamplingDistr = samplingDist->GetSamplingDistribution();
  SetSampleWeights(samplingDist);

  // add option "SAME" if necessary
  TString options(drawOptions);
  options.ToUpper();
  if(!options.Contains("SAME")) options.Append("SAME");
  if(!options.Contains("E1")) options.Append("E1");

  const Double_t xlow = *(std::min_element(fSamplingDistr.begin(),fSamplingDistr.end()));
  const Double_t xup  = *(std::max_element(fSamplingDistr.begin(),fSamplingDistr.end()));

  fhist = new TH1F(samplingDist->GetName(),samplingDist->GetTitle(),fbins,xlow,xup);

  fhist->GetXaxis()->SetTitle(samplingDist->GetVarName().Data());

  fVarName = samplingDist->GetVarName().Data();

  std::vector<Double_t>::iterator valuesIt = fSamplingDistr.begin();

  for(int w_idx = 0; valuesIt != fSamplingDistr.end(); ++valuesIt, ++w_idx)
    {
      if(fIsWeighted) fhist->Fill(*valuesIt,fSampleWeights[w_idx]);
      else fhist->Fill(*valuesIt);
    }

  fhist->Sumw2() ;

  //some basic aesthetics
  fhist->SetMarkerStyle(fMarkerType);
  fhist->SetMarkerColor(fColor);
  fhist->SetLineColor(fColor);

  fMarkerType++;
  fColor++;

  fhist->SetStats(kFALSE);

  addObject(fhist,options.Data());

  return;
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

//_____________________________________________________________________________
void SamplingDistPlot::Draw(const Option_t * /*options */ ) 
{
  // Draw this plot and all of the elements it contains. The specified options
  // only apply to the drawing of our frame. The options specified in our add...()
  // methods will be used to draw each object we contain.

  Float_t theMin(0.), theMax(0.), theYMax(0.);

  GetAbsoluteInterval(theMin,theMax,theYMax);

  RooRealVar xaxis("xaxis",fVarName.Data(),theMin,theMax);
  RooPlot* frame = xaxis.frame();
  frame->SetTitle("");
  frame->SetMaximum(theYMax);

  fIterator->Reset();
  TH1F *obj = 0;
  while((obj= (TH1F*)fIterator->Next()))
    //obj->Draw(fIterator->GetOption());
    frame->addTH1(obj,fIterator->GetOption());

  frame->Draw();

  return;
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
void SamplingDistPlot::SetLineColor(const Color_t color, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fhist->SetLineColor(color);
  }
  else{
    fIterator->Reset();
    TH1F *obj = 0;
    while((obj = (TH1F*)fIterator->Next())) {
      if(!strcmp(obj->GetName(),samplDist->GetName())){
	obj->SetLineColor(color);
	break;
      }
    }
  }

  return;
}

//_____________________________________________________________________________
void SamplingDistPlot::SetLineWidth(const Width_t lwidth, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fhist->SetLineWidth(lwidth);
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
void SamplingDistPlot::SetLineStyle(const Style_t style, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fhist->SetLineStyle(style);
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
void SamplingDistPlot::SetMarkerStyle(const Style_t style, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fhist->SetMarkerStyle(style);
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
void SamplingDistPlot::SetMarkerColor(const Color_t color, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fhist->SetMarkerColor(color);
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
void SamplingDistPlot::SetMarkerSize(const Size_t size, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fhist->SetMarkerSize(size);
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
void SamplingDistPlot::RebinDistribution(const Int_t rebinFactor, const SamplingDistribution *samplDist)
{
  if(samplDist == 0){
    fhist->Rebin(rebinFactor);
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
