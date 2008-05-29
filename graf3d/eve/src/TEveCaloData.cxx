// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveCaloData.h"

#include "TAxis.h"
#include "THStack.h"
#include "TH2.h"
#include "TMath.h"
#include "TList.h"


//______________________________________________________________________________
//
//  A central manager for calorimeter event data. It provides a list of
//  cells within requested phi and etha rng.
//

ClassImp(TEveCaloData);

TEveCaloData::TEveCaloData():TEveRefCnt(),
  fThreshold(0.001f)
{
   // Constructor.

}

//______________________________________________________________________________
void TEveCaloData::CellData_t::Configure(Float_t v, Float_t e1, Float_t e2, Float_t p1, Float_t p2)
{
   // Set parameters from cell ID.

   using namespace TMath;

   if (e1<0)
   {
      fThetaMin = Pi() - 2*ATan(Exp(- Abs(e2)));
      fThetaMax = Pi() - 2*ATan(Exp(- Abs(e1)));
      fZSideSign = -1;
   }
   else
   {
      fThetaMax = 2*ATan(Exp( -Abs(e1)));
      fThetaMin = 2*ATan(Exp( -Abs(e2)));
      fZSideSign = 1;
   }


   fEtaMin = e1;
   fEtaMax = e2;

   fPhiMin = p1;
   fPhiMax = p2;

   fValue  = v;
}

//______________________________________________________________________________
void TEveCaloData::CellData_t::Dump() const
{
  // Print member data.

   printf(">> theta %2.1f phi %2.1f val %2.2f \n",
          Theta()*TMath::RadToDeg(),
          Phi()*TMath::RadToDeg(), Value());
}

//______________________________________________________________________________
//
// A central manager for calorimeter data of an event written in TH2F.
// X axis present eta bin, Y axis present phi bin.
//

ClassImp(TEveCaloDataHist);


TEveCaloDataHist::TEveCaloDataHist(): fHStack(0)
{
   // Constructor.

   fHStack = new THStack();
}

TEveCaloDataHist::~TEveCaloDataHist()
{
   // Destructor.

   delete fHStack;
}

//______________________________________________________________________________
Int_t TEveCaloDataHist::GetCellList(Float_t minVal, Float_t maxVal,
                                    Float_t eta, Float_t etaD,
                                    Float_t phi, Float_t phiD,
                                    TEveCaloData::vCellId_t &out)

{
   // Get list of cell IDs in given eta and phi range.

   using namespace TMath;

   Float_t phiMin = phi - phiD;
   Float_t phiMax = phi + phiD;
   Float_t etaMin = eta - etaD;
   Float_t etaMax = eta + etaD;

   TH2F *hist0 = (TH2F*)fHStack->GetStack()->At(0);
   TAxis *ax = hist0->GetXaxis();
   TAxis *ay = hist0->GetYaxis();
   TH2F *hist;

   Int_t bin = 0;
   Float_t val = 0;
   Double_t phi0, phi1;
   Bool_t phiInside;
   for (Int_t ieta=1; ieta<=ax->GetNbins(); ieta++) {
      for (Int_t iphi=1; iphi<=ay->GetNbins(); iphi++)  {
         if ( ax->GetBinLowEdge(ieta)>=etaMin && ax->GetBinUpEdge(ieta)<=etaMax)
         {
            phiInside = kFALSE;
            phi0 = ay->GetBinLowEdge(iphi);
            phi1 = ay->GetBinUpEdge(iphi);

            if ( phi0>=phiMin && phi1 <=phiMax)
            {
               phiInside = kTRUE;
            }
            else if (phiMax>Pi() && phi1<phiMin)
            {
               phi0 += TwoPi();
               phi1 += TwoPi();
               if (phi0>phiMin && phi1<phiMax)
                  phiInside = kTRUE;
            }
            else if (phiMin<-Pi() && phi0>phiMax)
            {
               phi0 -= TwoPi();
               phi1 -= TwoPi();
               if (phi0>=phiMin && phi1<=phiMax)
                  phiInside = kTRUE;
            }

            if (phiInside)
            {
               TIter next(fHStack->GetHists());
               Int_t slice = 0;
               bin = hist0->GetBin(ieta, iphi);
               while ((hist = (TH2F*) next()) != 0) {
                  val = hist->GetBinContent(bin);
                  if (val>fThreshold && val>minVal && val<=maxVal)
                  {
                     out.push_back(TEveCaloData::CellId_t(bin, slice));
                  }
                  slice++;
               }
            }
         }
      }
   }
   return out.size();
}

//______________________________________________________________________________
void TEveCaloDataHist::GetCellData(const TEveCaloData::CellId_t &id, TEveCaloData::CellData_t& cellData)
{
  // Get cell geometry and value from cell ID.

   TH2F* hist  = (TH2F*) (fHStack->GetHists()->At(id.fSlice));

   Int_t x, y, z;
   hist->GetBinXYZ(id.fTower, x, y, z);

   cellData.Configure(hist->GetBinContent(id.fTower),
                      hist->GetXaxis()->GetBinLowEdge(x),
                      hist->GetXaxis()->GetBinUpEdge(x),
                      hist->GetYaxis()->GetBinLowEdge(y),
                      hist->GetYaxis()->GetBinUpEdge(y));
}

//______________________________________________________________________________
void TEveCaloDataHist::GetCellData(const TEveCaloData::CellId_t &id, Float_t phi, Float_t phiRng,
                                   TEveCaloData::CellData_t& cellData)
{
   // Get cell geometry and value from cell ID.
   // Respect external phi range shifted for a given phi.

   using namespace TMath;

   Float_t phiMin = phi-phiRng;
   Float_t phiMax = phi+phiRng;

   TH2F* hist  = (TH2F*) (fHStack->GetHists()->At(id.fSlice));

   Int_t x, y, z;
   hist->GetBinXYZ(id.fTower, x, y, z);

   Float_t phi1 = hist->GetYaxis()->GetBinLowEdge(y);
   Float_t phi2 = hist->GetYaxis()->GetBinUpEdge(y);

   if (phiMax>Pi() && phi2<=phiMin)
   {
      phi1 += TwoPi();
      phi2 += TwoPi();
   }
   else if (phiMin<-Pi() && phi1>=phiMax)
   {
      phi1 -= TwoPi();
      phi2 -= TwoPi();
   }

   cellData.Configure(hist->GetBinContent(id.fTower),
                      hist->GetXaxis()->GetBinLowEdge(x),
                      hist->GetXaxis()->GetBinUpEdge(x),
                      phi1, phi2);
}

//______________________________________________________________________________
void TEveCaloDataHist::AddHistogram(TH2F* h)
{
   // Add  new slice to calo tower.

   fHStack->Add(h);
}

//______________________________________________________________________________
Int_t TEveCaloDataHist::GetNSlices() const
{
   // Get number of tower slices.

   return fHStack->GetHists()->GetEntries();
}

//______________________________________________________________________________
Float_t TEveCaloDataHist::GetMaxVal() const
{
   // Returns the maximum of all added histograms.

   return fHStack->GetMaximum();
}

//______________________________________________________________________________
void TEveCaloDataHist::GetEtaLimits(Double_t &min, Double_t &max) const
{
   // Get eta limits.

   TH2F* hist  = (TH2F*) (fHStack->GetHists()->At(0));
   min = hist->GetXaxis()->GetXmin();
   max = hist->GetXaxis()->GetXmax();
}

//______________________________________________________________________________
void TEveCaloDataHist::GetPhiLimits(Double_t &min, Double_t &max) const
{
   // Get phi limits.

   TH2F* hist  = (TH2F*) (fHStack->GetHists()->At(0));
   min = hist->GetYaxis()->GetXmin();
   max = hist->GetYaxis()->GetXmax();
}

//______________________________________________________________________________
TAxis* TEveCaloDataHist::GetEtaBins()
{
   // Get eta axis.

   TH2F* hist  = (TH2F*) (fHStack->GetHists()->At(0));
   return hist->GetXaxis();
}

//______________________________________________________________________________
TAxis* TEveCaloDataHist::GetPhiBins()
{
   // Get phi axis.

   TH2F* hist  = (TH2F*) (fHStack->GetHists()->At(0));
   return hist->GetYaxis();
}

//______________________________________________________________________________
const TH2F* TEveCaloDataHist::GetHistogram(Int_t slice)
{
   // Get histogram for given slice.

   if (slice > GetNSlices())
      return 0;

   return (TH2F*)fHStack->GetHists()->At(slice);
}
