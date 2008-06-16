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
#include "TEveCalo.h"

#include "TAxis.h"
#include "THStack.h"
#include "TH2.h"
#include "TMath.h"
#include "TList.h"


//==============================================================================
// TEveCaloData
//==============================================================================

//______________________________________________________________________________
//
//  A central manager for calorimeter event data. It provides a list of
//  cells within requested phi and eta range.
//

ClassImp(TEveCaloData);

//______________________________________________________________________________
TEveCaloData::TEveCaloData():
   TEveRefBackPtr(),

   fEtaAxis(0),
   fPhiAxis(0)
{
   // Constructor.
}

//______________________________________________________________________________
void TEveCaloData::SetSliceThreshold(Int_t slice, Float_t val)
{
   // Set threshold for given slice.

   fSliceInfos[slice].fThreshold = val;
   InvalidateUsersCellIdCache();
}

//______________________________________________________________________________
void TEveCaloData::SetSliceColor(Int_t slice, Color_t col)
{
   // Set color for given slice.
   
   fSliceInfos[slice].fColor = col;
   StampBackPtrElements(TEveElement::kCBObjProps);
}

//______________________________________________________________________________
void TEveCaloData::InvalidateUsersCellIdCache()
{
   // Invalidate cell ids cache on back ptr references.

   TEveCaloViz* calo;
   std::list<TEveElement*>::iterator i = fBackRefs.begin();
   while (i != fBackRefs.end())
   {
      calo = dynamic_cast<TEveCaloViz*>(*i);
      calo->InvalidateCellIdCache();
      calo->StampObjProps();
      ++i;
   }
}

//______________________________________________________________________________
void TEveCaloData::DataChanged()
{
   // Tell users (TEveCaloViz instances using this data) that data
   // has changed and they should update the limits/scales etc.
   // This is done by calling TEveCaloViz::DataChanged().

   TEveCaloViz* calo;
   std::list<TEveElement*>::iterator i = fBackRefs.begin();
   while (i != fBackRefs.end())
   {
      calo = dynamic_cast<TEveCaloViz*>(*i);
      calo->DataChanged();
      calo->StampObjProps();
      ++i;
   }
}

/**************************************************************************/

//______________________________________________________________________________
Float_t TEveCaloData::CellData_t::Value(Bool_t isEt) const
{
   // Return energy value associated with the cell, usually Et.
   // If isEt is false it is transformed into energy E.

   if (isEt)
      return fValue;
   else
      return TMath::Abs(fValue/TMath::Cos(Theta()));
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

   printf(">> theta %2.1f phi %2.1f Et %2.2f E  %2.2f \n",
          Theta()*TMath::RadToDeg(),
          Phi()*TMath::RadToDeg(), Value(kFALSE), Value(kTRUE));
}


//==============================================================================
// TEveCaloDataHist
//==============================================================================

//______________________________________________________________________________
//
// A central manager for calorimeter data of an event written in TH2F.
// X axis is used for eta and Y axis for phi.
//

ClassImp(TEveCaloDataHist);


TEveCaloDataHist::TEveCaloDataHist():
   TEveCaloData(),

   fHStack(0),
   fMaxValEt(0),
   fMaxValE(0)
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
void TEveCaloDataHist::DataChanged()
{ 
   // Update limits and notify data users. 

   using namespace TMath;

   // update max E/Et values
   fMaxValE = 0;
   fMaxValEt = 0;

   if (fHStack->GetHists()->First())
   {  
      TH2 *ah = (TH2*)fHStack->GetHists()->First();
      fEtaAxis = ah->GetXaxis();
      fPhiAxis = ah->GetYaxis();

      Int_t bin;
      Double_t value, cos, eta;
      TH2 *stack =  (TH2*)fHStack->GetStack()->Last();
      for (Int_t ieta=1; ieta<=fEtaAxis->GetNbins(); ieta++) 
      {
         eta = fEtaAxis->GetBinCenter(ieta); // conversion E/Et
         for (Int_t iphi=1; iphi<=fPhiAxis->GetNbins(); iphi++)  
         {
            bin = stack->GetBin(ieta, iphi);
            value = stack->GetBinContent(bin);

            if (value > fMaxValEt ) fMaxValEt = value;

            cos = Cos(2*ATan(Exp( -Abs(eta))));
            value /= Abs(cos);
            if (value > fMaxValE) fMaxValE = value;
         }
      }
   }

   TEveCaloData::DataChanged();
} 

//______________________________________________________________________________
void TEveCaloDataHist::GetCellList(Float_t eta, Float_t etaD,
                                   Float_t phi, Float_t phiD,
                                   TEveCaloData::vCellId_t &out) const
{
   // Get list of cell IDs in given eta and phi range.

   using namespace TMath;

   Float_t etaMin = eta - etaD*0.5; 
   Float_t etaMax = eta + etaD*0.5; 

   Float_t phiMin = phi - phiD*0.5; 
   Float_t phiMax = phi + phiD*0.5; 

   Int_t nEta = fEtaAxis->GetNbins();
   Int_t nPhi = fPhiAxis->GetNbins();
   Int_t nSlices = GetNSlices();

   TH2* h0 = fSliceInfos[0].fHist;
   Int_t bin = 0;

   for (Int_t ieta=1; ieta<=nEta; ieta++) 
   {
      if (fEtaAxis->GetBinLowEdge(ieta) >= etaMin && fEtaAxis->GetBinUpEdge(ieta) <= etaMax)
      {
         for (Int_t iphi=1; iphi<=nPhi; iphi++)
         {
            if (TEveUtil::IsU1IntervalOverlappingByMinMax
                (phiMin+1e-5, phiMax-1e-5, fPhiAxis->GetBinLowEdge(iphi), fPhiAxis->GetBinUpEdge(iphi)))            
            {  
               bin = h0->GetBin(ieta, iphi);
               for (Int_t s=0; s<nSlices; s++)
               {
                  if (fSliceInfos[s].fHist->GetBinContent(bin) > fSliceInfos[s].fThreshold )
                     out.push_back(TEveCaloData::CellId_t(bin, s));
               } // hist slices
            }
         } // phi bins
      }
   } // eta bins
}

//______________________________________________________________________________
void TEveCaloDataHist::GetCellData(const TEveCaloData::CellId_t &id,
                                   TEveCaloData::CellData_t& cellData) const
{
  // Get cell geometry and value from cell ID.

   TH2F* hist  = fSliceInfos[id.fSlice].fHist;

   Int_t x, y, z;
   hist->GetBinXYZ(id.fTower, x, y, z);

   cellData.Configure(hist->GetBinContent(id.fTower),
                      hist->GetXaxis()->GetBinLowEdge(x),
                      hist->GetXaxis()->GetBinUpEdge(x),
                      hist->GetYaxis()->GetBinLowEdge(y),
                      hist->GetYaxis()->GetBinUpEdge(y));
}

//______________________________________________________________________________
void TEveCaloDataHist::GetCellData(const TEveCaloData::CellId_t &id,
                                   Float_t phi, Float_t phiRng,
                                   TEveCaloData::CellData_t& cellData) const
{
   // Get cell geometry and value from cell ID.
   // Respect external phi range shifted for a given phi.

   using namespace TMath;

   Float_t phiMin = phi-phiRng;
   Float_t phiMax = phi+phiRng;

   TH2F* hist  = fSliceInfos[id.fSlice].fHist;

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
Int_t TEveCaloDataHist::AddHistogram(TH2F* hist)
{
   // Add new slice to calo tower. Updates cached variables fMaxValE
   // and fMaxValEt
   // Return last index in the vector of slice infos.

   fHStack->Add(hist);

   Int_t id = fSliceInfos.size();
   fSliceInfos.push_back(SliceInfo_t(hist));
   fSliceInfos[id].fName = hist->GetName();
   fSliceInfos[id].fColor = hist->GetLineColor();
   fSliceInfos[id].fID = id;

   DataChanged();
 
   return id;
}

//______________________________________________________________________________
Int_t TEveCaloDataHist::GetNSlices() const
{
   // Get number of tower slices.

   return fHStack->GetHists()->GetSize();
}

//______________________________________________________________________________
void TEveCaloDataHist::GetEtaLimits(Double_t &min, Double_t &max) const
{
   // Get eta limits.

   min = fEtaAxis->GetXmin();
   max = fEtaAxis->GetXmax();
}

//______________________________________________________________________________
void TEveCaloDataHist::GetPhiLimits(Double_t &min, Double_t &max) const
{
   // Get phi limits.

   min = fPhiAxis->GetXmin();
   max = fPhiAxis->GetXmax();
}
