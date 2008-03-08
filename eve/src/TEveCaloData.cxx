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
#include "TObjArray.h"
#include "TH2.h"
#include "TMath.h"


//______________________________________________________________________________
//
//  A central manager for calorimeter event data. It provides a list of
//  cells within requested phi and etha rng. 
//

ClassImp(TEveCaloData);

TEveCaloData::TEveCaloData():TEveRefCnt()
{
   // Constructor.

}

//______________________________________________________________________________
void TEveCaloData::CellData_t::Configure(Float_t v, Float_t e1, Float_t e2, Float_t p1, Float_t p2)
{
   // Set parameters from cell ID.

   fValue  = v;
   fPhiMin = p1;
   fPhiMax = p2;

   fThetaMax = 2*TMath::ATan(TMath::Exp(-(TMath::Abs(e1))));
   fThetaMin = 2*TMath::ATan(TMath::Exp(-(TMath::Abs(e2))));

   fZSideSign = (e1 > 0) ? 1:-1;
}

//______________________________________________________________________________
Float_t TEveCaloData::CellData_t::ThetaMin(Bool_t sig) const
{
   // Get minimum theta in radians. By default returns value calculated from
   // absolute vale of eta.

   if (sig && fZSideSign == -1) return TMath::Pi() - fThetaMin;
   return fThetaMin;
}

//______________________________________________________________________________
Float_t TEveCaloData::CellData_t::ThetaMax(Bool_t sig) const
{
   // Get maximum theta in radians. By default returns value calculated from
   // absolute vale of eta.

   if (sig && fZSideSign == -1 ) return TMath::Pi() - fThetaMax;
   return fThetaMax;
}

//______________________________________________________________________________
void TEveCaloData::CellData_t::Dump() const
{
   printf(">> theta %2.1f phi %2.1f val %2.2f \n",
          Theta(kTRUE)*TMath::RadToDeg(),
          Phi()*TMath::RadToDeg(), Value());
}

//______________________________________________________________________________
//
// A central manager for calorimeter data of an event written in TH2F.
// X axis present eta bin, Y axis phi bin.
//

ClassImp(TEveCaloDataHist);


TEveCaloDataHist::TEveCaloDataHist(): fHStack(0)
{
   // Constructor.

   fHStack = new TObjArray();
}

TEveCaloDataHist::~TEveCaloDataHist()
{
   // Destructor.

   delete fHStack;
}

//______________________________________________________________________________
Int_t TEveCaloDataHist::GetCellList(Float_t eta, Float_t etaD,
                                    Float_t phi, Float_t phiD,
                                    Float_t threshold, TEveCaloData::vCellId_t &out)

{
   using namespace TMath;

   etaD *= 1.01f;
   phiD *= 1.01f;

   // eta interval
   Float_t etaMin = eta - etaD*0.5f;
   Float_t etaMax = eta + etaD*0.5f;

   // phi interval
   Float_t pr[4];
   Float_t phi1  = phi - phiD;
   Float_t phi2  = phi + phiD;
   if (phi2 >TMath::Pi() && phi1>-Pi()) {
      pr[0] =  phi1;
      pr[1] =  Pi();
      pr[2] =  -Pi();
      pr[3] =  -TwoPi()+phi2;
      //printf("more than +++Pi \n");
   }
   else if (phi1<-TMath::Pi() && phi2<=Pi()) {
      pr[0] = -Pi();
      pr[1] =  phi2;
      pr[2] =  TwoPi()+phi1;
      pr[3] =  Pi();
      // printf("less  than ---Pi \n");
   }
   else {
      pr[0] = pr[2] = phi1;
      pr[1] = pr[3] = phi2;
      // printf("natural \n");
   }

   TH2F *hist0 = (TH2F*)fHStack->At(0);
   TAxis *ax = hist0->GetXaxis();
   TAxis *ay = hist0->GetYaxis();
   TH2F *hist;

   for (Int_t ieta=0; ieta<ay->GetNbins(); ieta++) {
      for (Int_t iphi=0; iphi<ax->GetNbins(); iphi++)  {
         //printf("compare eta (%f, %f) (%f, %f)\n", ax->GetBinLowEdge(ieta),ax->GetBinUpEdge(ieta), etaMin,etaMax );
         if (   ax->GetBinLowEdge(ieta)   >= etaMin
                && ax->GetBinUpEdge(ieta) <  etaMax
                && ((ay->GetBinLowEdge(iphi)>=pr[0] && ay->GetBinUpEdge(iphi)<pr[1])
                    || (ay->GetBinLowEdge(iphi)>=pr[2] && ay->GetBinUpEdge(iphi)<pr[3])))
         {

            Int_t bin = hist0->GetBin(ieta, iphi);
            TIter next(fHStack);
            Int_t slice = 0;
            while ((hist = (TH2F*) next()) != 0) {
               // printf("add cell %d %d \n",bin, slice);
               if (hist->GetBinContent(bin) > threshold)
                  out.push_back(TEveCaloData::CellId_t(bin, slice));
               slice++;
            }
         }
      }
   }
   return out.size();
}

//______________________________________________________________________________
void TEveCaloDataHist::GetCellData(const TEveCaloData::CellId_t &id, TEveCaloData::CellData_t& cellData)
{
   TH2F* hist  = (TH2F*) (fHStack->At(id.fSlice));

   Int_t x, y, z;
   hist->GetBinXYZ(id.fTower, x, y, z);
   cellData.Configure(hist->GetBinContent(id.fTower),
                      hist->GetXaxis()->GetBinLowEdge(x),
                      hist->GetXaxis()->GetBinUpEdge(x),
                      hist->GetYaxis()->GetBinLowEdge(y),
                      hist->GetYaxis()->GetBinUpEdge(y));

   // printf("GetCellData eta %d, phi%d ", hist->GetYaxis()->GetBinLowEdge(y), hist->GetXaxis()->GetBinLowEdge(x));
   //   cellData.Dump();
}

//______________________________________________________________________________
void TEveCaloDataHist::AddHistogram(TH2F* h)
{
   // Add  ne slice to calo tower.

   fHStack->Add(h);
}

//______________________________________________________________________________
Int_t TEveCaloDataHist::GetNSlices() const
{
   // Get number of tower slices.

   return fHStack->GetEntriesFast();
}

//______________________________________________________________________________
const TAxis* TEveCaloDataHist::GetEtaBins()
{
   // Get eta axis.

   TH2F* hist  = (TH2F*) (fHStack->At(0));
   return hist->GetXaxis();
}

//______________________________________________________________________________
const TAxis* TEveCaloDataHist::GetPhiBins()
{
   // Get phi axis.

   TH2F* hist  = (TH2F*) (fHStack->At(0));
   return hist->GetYaxis();
}

//______________________________________________________________________________
const TH2F* TEveCaloDataHist::GetHistogram(Int_t slice)
{
   // Get histogram for given slice.

   if (slice > GetNSlices())
      return 0;

   return (TH2F*)fHStack->At(slice);
}
