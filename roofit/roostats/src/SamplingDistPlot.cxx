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

/** \class RooStats::SamplingDistPlot
    \ingroup Roostats

This class provides simple and straightforward utilities to plot SamplingDistribution
objects.
*/

#include "RooStats/SamplingDistPlot.h"

#include "RooStats/SamplingDistribution.h"

#include "RooRealVar.h"
#include "TStyle.h"
#include "TLine.h"
#include "TFile.h"
#include "TVirtualPad.h"  // for gPad

#include <algorithm>
#include <iostream>


#include "RooMsgService.h"

#include "TMath.h"


using namespace RooStats;
using std::string;

////////////////////////////////////////////////////////////////////////////////
/// SamplingDistPlot default constructor with bin size

SamplingDistPlot::SamplingDistPlot(Int_t nbins) : fBins{nbins} {}

////////////////////////////////////////////////////////////////////////////////
/// SamplingDistPlot constructor with bin size, min and max

SamplingDistPlot::SamplingDistPlot(Int_t nbins, double min, double max) : fBins{nbins}
{
   SetXRange(min, max);
}

////////////////////////////////////////////////////////////////////////////////
/// destructors - delete objects contained in the list

SamplingDistPlot::~SamplingDistPlot()
{
   fItems.Delete();
   fOtherItems.Delete();
   if (fRooPlot) delete fRooPlot;
}


////////////////////////////////////////////////////////////////////////////////
/// adds sampling distribution (and normalizes if "NORMALIZE" is given as an option)

double SamplingDistPlot::AddSamplingDistribution(const SamplingDistribution *samplingDist, Option_t *drawOptions) {
   fSamplingDistr = samplingDist->GetSamplingDistribution();
   if( fSamplingDistr.empty() ) {
      coutW(Plotting) << "Empty sampling distribution given to plot. Skipping." << std::endl;
      return 0.0;
   }
   SetSampleWeights(samplingDist);

   TString options(drawOptions);
   options.ToUpper();

   double xmin(TMath::Infinity());
   double xmax(-TMath::Infinity());
   // remove cases where xmin and xmax are +/- inf
   for( unsigned int i=0; i < fSamplingDistr.size(); i++ ) {
      if( fSamplingDistr[i] < xmin  &&  fSamplingDistr[i] != -TMath::Infinity() ) {
         xmin = fSamplingDistr[i];
      }
      if( fSamplingDistr[i] > xmax  &&  fSamplingDistr[i] != TMath::Infinity() ) {
         xmax = fSamplingDistr[i];
      }
   }
   if( xmin >= xmax ) {
      coutW(Plotting) << "Could not determine xmin and xmax of sampling distribution that was given to plot." << std::endl;
      xmin = -1.0;
      xmax = 1.0;
   }


   // add 1.5 bins left and right
   assert(fBins > 1);
   double binWidth = (xmax-xmin)/(fBins);
   double xlow = xmin - 1.5*binWidth;
   double xup  = xmax + 1.5*binWidth;
   if( !TMath::IsNaN(fXMin) ) xlow = fXMin;
   if( !TMath::IsNaN(fXMax) ) xup = fXMax;

   fHist = new TH1F(samplingDist->GetName(), samplingDist->GetTitle(), fBins, xlow, xup);
   fHist->SetDirectory(nullptr);  // make the object managed by this class

   if( fVarName.Length() == 0 ) fVarName = samplingDist->GetVarName();
   fHist->GetXaxis()->SetTitle(fVarName.Data());


   std::vector<double>::iterator valuesIt = fSamplingDistr.begin();
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

   fHist->SetStats(false);

   addObject(fHist, options.Data());

   TString title = samplingDist->GetTitle();
   if(fLegend  &&  title.Length() > 0) fLegend->AddEntry(fHist, title, "L");

   return 1./weightSum;
}

////////////////////////////////////////////////////////////////////////////////

double SamplingDistPlot::AddSamplingDistributionShaded(const SamplingDistribution *samplingDist, double minShaded, double maxShaded, Option_t *drawOptions) {
   if( samplingDist->GetSamplingDistribution().empty() ) {
      coutW(Plotting) << "Empty sampling distribution given to plot. Skipping." << std::endl;
      return 0.0;
   }
   double scaleFactor = AddSamplingDistribution(samplingDist, drawOptions);

   TH1F *shaded = static_cast<TH1F*>(fHist->Clone((string(samplingDist->GetName())+string("_shaded")).c_str()));
   shaded->SetDirectory(nullptr);
   shaded->SetFillStyle(fFillStyle++);
   shaded->SetLineWidth(1);

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

////////////////////////////////////////////////////////////////////////////////

void SamplingDistPlot::AddLine(double x1, double y1, double x2, double y2, const char* title) {
   TLine *line = new TLine(x1, y1, x2, y2);
   line->SetLineWidth(3);
   line->SetLineColor(kBlack);

   if(fLegend  &&  title) fLegend->AddEntry(line, title, "L");

   addOtherObject(line, ""); // no options
}

////////////////////////////////////////////////////////////////////////////////
/// add an histogram (it will be cloned);

void SamplingDistPlot::AddTH1(TH1* h, Option_t *drawOptions) {
   if(fLegend  &&  h->GetTitle()) fLegend->AddEntry(h, h->GetTitle(), "L");
   TH1 * hcopy = static_cast<TH1*>(h->Clone());
   hcopy->SetDirectory(nullptr);
   addObject(hcopy, drawOptions);
}
void SamplingDistPlot::AddTF1(TF1* f, const char* title, Option_t *drawOptions) {
   if(fLegend  &&  title) fLegend->AddEntry(f, title, "L");
   addOtherObject(f->Clone(), drawOptions);
}

////////////////////////////////////////////////////////////////////////////////
///Determine if the sampling distribution has weights and store them

void SamplingDistPlot::SetSampleWeights(const SamplingDistribution* samplingDist)
{
  fIsWeighted = false;

  if(!samplingDist->GetSampleWeights().empty()){
    fIsWeighted = true;
    fSampleWeights = samplingDist->GetSampleWeights();
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a generic object to this plot. The specified options will be
/// used to Draw() this object later. The caller transfers ownership
/// of the object with this call, and the object will be deleted
/// when its containing plot object is destroyed.

void SamplingDistPlot::addObject(TObject *obj, Option_t *drawOptions)
{

  if(nullptr == obj) {
    std::cerr << fName << "::addObject: called with a null pointer" << std::endl;
    return;
  }

  fItems.Add(obj,drawOptions);

  return;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a generic object to this plot. The specified options will be
/// used to Draw() this object later. The caller transfers ownership
/// of the object with this call, and the object will be deleted
/// when its containing plot object is destroyed.

void SamplingDistPlot::addOtherObject(TObject *obj, Option_t *drawOptions)
{
  if(nullptr == obj) {
     coutE(InputArguments) << fName << "::addOtherObject: called with a null pointer" << std::endl;
     return;
  }

  fOtherItems.Add(obj,drawOptions);

  return;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw this plot and all of the elements it contains. The specified options
/// only apply to the drawing of our frame. The options specified in our add...()
/// methods will be used to draw each object we contain.

void SamplingDistPlot::Draw(Option_t * /*options */) {
   ApplyDefaultStyle();

   double theMin(0.);
   double theMax(0.);
   double theYMin(std::numeric_limits<float>::quiet_NaN());
   double theYMax(0.);
   GetAbsoluteInterval(theMin, theMax, theYMax);
   if( !TMath::IsNaN(fXMin) ) theMin = fXMin;
   if( !TMath::IsNaN(fXMax) ) theMax = fXMax;
   if( !TMath::IsNaN(fYMin) ) theYMin = fYMin;
   if( !TMath::IsNaN(fYMax) ) theYMax = fYMax;

   RooRealVar xaxis("xaxis", fVarName.Data(), theMin, theMax);

   //L.M. by drawing many times we create a memory leak ???
   if (fRooPlot) delete fRooPlot;

   bool dirStatus = RooPlot::addDirectoryStatus();
   // make the RooPlot managed by this class
   RooPlot::setAddDirectoryStatus(false);
   fRooPlot = xaxis.frame();
   RooPlot::setAddDirectoryStatus(dirStatus);
   if (!fRooPlot) {
     coutE(InputArguments) << "invalid variable to plot" << std::endl;
     return;
   }
   fRooPlot->SetTitle("");
   if( !TMath::IsNaN(theYMax) ) {
      //coutI(InputArguments) << "Setting maximum to " << theYMax << std::endl;
      fRooPlot->SetMaximum(theYMax);
   }
   if( !TMath::IsNaN(theYMin) ) {
      //coutI(InputArguments) << "Setting minimum to " << theYMin << std::endl;
      fRooPlot->SetMinimum(theYMin);
   }

   for(auto * obj : static_range_cast<TH1F*>(fItems)) {
      //obj->Draw(fIterator->GetOption());
      // add cloned objects to avoid mem leaks
      TH1 * cloneObj = static_cast<TH1*>(obj->Clone());
      if( !TMath::IsNaN(theYMax) ) {
         //coutI(InputArguments) << "Setting maximum of TH1 to " << theYMax << std::endl;
         cloneObj->SetMaximum(theYMax);
      }
      if( !TMath::IsNaN(theYMin) ) {
         //coutI(InputArguments) << "Setting minimum of TH1 to " << theYMin << std::endl;
         cloneObj->SetMinimum(theYMin);
      }
      cloneObj->SetDirectory(nullptr);  // transfer ownership of the object
      fRooPlot->addTH1(cloneObj, obj->GetOption());
   }

   for(TObject * otherObj : fOtherItems) {
      TObject * cloneObj = otherObj->Clone();
      fRooPlot->addObject(cloneObj, otherObj->GetOption());
   }


   if(fLegend) fRooPlot->addObject(fLegend);

   if(bool(gStyle->GetOptLogx()) != fLogXaxis) {
      if(!fApplyStyle) coutW(Plotting) << "gStyle will be changed to adjust SetOptLogx(...)" << std::endl;
      gStyle->SetOptLogx(fLogXaxis);
   }
   if(bool(gStyle->GetOptLogy()) != fLogYaxis) {
      if(!fApplyStyle) coutW(Plotting) << "gStyle will be changed to adjust SetOptLogy(...)" << std::endl;
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

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

void SamplingDistPlot::GetAbsoluteInterval(double &theMin, double &theMax, double &theYMax) const
{
   double tmpmin = TMath::Infinity();
   double tmpmax = -TMath::Infinity();
   double tmpYmax = -TMath::Infinity();

  for(auto * obj : static_range_cast<TH1F*>(fItems)) {
    if(obj->GetXaxis()->GetXmin() < tmpmin) tmpmin = obj->GetXaxis()->GetXmin();
    if(obj->GetXaxis()->GetXmax() > tmpmax) tmpmax = obj->GetXaxis()->GetXmax();
    if(obj->GetMaximum() > tmpYmax) tmpYmax = obj->GetMaximum() + 0.1*obj->GetMaximum();
  }

  theMin = tmpmin;
  theMax = tmpmax;
  theYMax = tmpYmax;

  return;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets line color for given sampling distribution and
/// fill color for the associated shaded TH1F.

void SamplingDistPlot::SetLineColor(Color_t color, const SamplingDistribution *sampleDist) {
   if (sampleDist == nullptr) {
      fHist->SetLineColor(color);

      TString shadedName(fHist->GetName());
      shadedName += "_shaded";

      for(auto * obj : static_range_cast<TH1F*>(fItems)) {
         if (!strcmp(obj->GetName(), shadedName.Data())) {
            obj->SetLineColor(color);
            obj->SetFillColor(color);
            //break;
         }
      }
   } else {

      TString shadedName(sampleDist->GetName());
      shadedName += "_shaded";

      for(auto * obj : static_range_cast<TH1F*>(fItems)) {
         if (!strcmp(obj->GetName(), sampleDist->GetName())) {
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

////////////////////////////////////////////////////////////////////////////////

void SamplingDistPlot::SetLineWidth(Width_t lwidth, const SamplingDistribution *sampleDist)
{
  if(sampleDist == nullptr){
    fHist->SetLineWidth(lwidth);
  }
  else{
    for(auto * obj : static_range_cast<TH1F*>(fItems)) {
      if(!strcmp(obj->GetName(),sampleDist->GetName())){
   obj->SetLineWidth(lwidth);
   break;
      }
    }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////

void SamplingDistPlot::SetLineStyle(Style_t style, const SamplingDistribution *sampleDist)
{
  if(sampleDist == nullptr){
    fHist->SetLineStyle(style);
  }
  else{
    for(auto * obj : static_range_cast<TH1F*>(fItems)) {
      if(!strcmp(obj->GetName(),sampleDist->GetName())){
   obj->SetLineStyle(style);
   break;
      }
    }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////

void SamplingDistPlot::SetMarkerStyle(Style_t style, const SamplingDistribution *sampleDist)
{
  if(sampleDist == nullptr){
    fHist->SetMarkerStyle(style);
  }
  else{
    for(auto * obj : static_range_cast<TH1F*>(fItems)) {
      if(!strcmp(obj->GetName(),sampleDist->GetName())){
   obj->SetMarkerStyle(style);
   break;
      }
    }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////

void SamplingDistPlot::SetMarkerColor(Color_t color, const SamplingDistribution *sampleDist)
{
  if(sampleDist == nullptr){
    fHist->SetMarkerColor(color);
  }
  else{
    for(auto * obj : static_range_cast<TH1F*>(fItems)) {
      if(!strcmp(obj->GetName(),sampleDist->GetName())){
   obj->SetMarkerColor(color);
   break;
      }
    }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////

void SamplingDistPlot::SetMarkerSize(Size_t size, const SamplingDistribution *sampleDist)
{
  if(sampleDist == nullptr){
    fHist->SetMarkerSize(size);
  }
  else{
    for(auto * obj : static_range_cast<TH1F*>(fItems)) {
      if(!strcmp(obj->GetName(),sampleDist->GetName())){
   obj->SetMarkerSize(size);
   break;
      }
    }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////

TH1F* SamplingDistPlot::GetTH1F(const SamplingDistribution *sampleDist)
{
  if(sampleDist == nullptr){
    return fHist;
  }else{
    for(auto * obj : static_range_cast<TH1F*>(fItems)) {
      if(!strcmp(obj->GetName(),sampleDist->GetName())){
        return obj;
      }
    }
  }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

void SamplingDistPlot::RebinDistribution(Int_t rebinFactor, const SamplingDistribution *sampleDist)
{
  if(sampleDist == nullptr){
    fHist->Rebin(rebinFactor);
  }
  else{
    for(auto * obj : static_range_cast<TH1F*>(fItems)) {
      if(!strcmp(obj->GetName(),sampleDist->GetName())){
   obj->Rebin(rebinFactor);
   break;
      }
    }
  }

  return;
}

////////////////////////////////////////////////////////////////////////////////
/// TODO test

void SamplingDistPlot::DumpToFile(const char* RootFileName, Option_t *option, const char *ftitle, Int_t compress) {
   // All the objects are written to rootfile

   if(!fRooPlot) {
      std::cout << "Plot was not drawn yet. Dump can only be saved after it was drawn with Draw()." << std::endl;
      return;
   }

   TFile ofile(RootFileName, option, ftitle, compress);
   ofile.cd();
   fRooPlot->Write();
   ofile.Close();
}
