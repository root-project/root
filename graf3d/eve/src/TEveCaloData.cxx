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

#include <cassert>
#include <algorithm>


//------------------------------------------------------------------------------
// TEveCaloData::CellGeom_t
//------------------------------------------------------------------------------


//______________________________________________________________________________
void TEveCaloData::CellGeom_t::Dump() const
{
   // Print member data.

   printf("%f, %f %f, %f \n", fEtaMin, fEtaMax, fPhiMin, fPhiMax);
}

//______________________________________________________________________________
void TEveCaloData::CellGeom_t::Configure(Float_t etaMin, Float_t etaMax, Float_t phiMin, Float_t phiMax)
{
   fEtaMin = etaMin;
   fEtaMax = etaMax;

   fPhiMin = phiMin;
   fPhiMax = phiMax;

   fThetaMin = EtaToTheta(fEtaMax);
   fThetaMax = EtaToTheta(fEtaMin);
}

//------------------------------------------------------------------------------
// TEveCaloData::CellData_t
//------------------------------------------------------------------------------

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
void TEveCaloData::CellData_t::Dump() const
{
   // Print member data.

   printf("%f, %f %f, %f \n", fEtaMin, fEtaMax, fPhiMin, fPhiMax);
}

//______________________________________________________________________________
Float_t* TEveCaloData::RebinData_t::GetSliceVals(Int_t bin)
{

   //   printf("get val vec bin %d size %d\n", bin, fBinData.size());
   if (fBinData[bin] == -1)
   {
      fBinData[bin] = fSliceData.size();

      for (Int_t i=0; i<fNSlices; i++)
         fSliceData.push_back(0.f);
   }

   return &fSliceData[fBinData[bin]];
}

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
TEveCaloData::TEveCaloData(const char* n, const char* t):
   TEveElement(),
   TNamed(n, t),

   fEtaAxis(0),
   fPhiAxis(0),

   fWrapTwoPi(kTRUE),

   fMaxValEt(0),
   fMaxValE(0),

   fEps(0)
{
   // Constructor.
}

//______________________________________________________________________________
void TEveCaloData::SelectElement(Bool_t s)
{
   // Virtual method TEveElement::SelectElement.
   // Clear selected towers when deselected.

   if (s == kFALSE)
      fCellsSelected.clear();

   TEveElement::SelectElement(s);
}

//______________________________________________________________________________
void TEveCaloData::FillImpliedSelectedSet(Set_t& impSelSet)
{
   // Populate set impSelSet with derived / dependant elements.
   //

   //TEveElement::FillImpliedSelectedSet(impSelSet);

   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      //printf("insert to imp selected %s \n", (*i)->GetElementName());
         impSelSet.insert(*i);
   }

}

//______________________________________________________________________________
void TEveCaloData::PrintCellsSelected()
{
   // Print selected cells info.

   printf("%d Selected selected cells:\n", (Int_t)fCellsSelected.size());
   CellData_t cellData;

   for (vCellId_i i = fCellsSelected.begin(); i != fCellsSelected.end(); ++i)
   {
      GetCellData(*i, cellData);
      printf("Tower [%d] Slice [%d] Value [%.2f] ", i->fTower, i->fSlice, cellData.fValue);
      printf("Eta:(%f, %f) Phi(%f, %f)\n",  cellData.fEtaMin, cellData.fEtaMax, cellData.fPhiMin, cellData.fPhiMax);
   }
}

//______________________________________________________________________________
void TEveCaloData::SetSliceThreshold(Int_t slice, Float_t val)
{
   // Set threshold for given slice.

   fSliceInfos[slice].fThreshold = val;
   InvalidateUsersCellIdCache();
}

//______________________________________________________________________________
Float_t TEveCaloData::GetSliceThreshold(Int_t slice) const
{
   // Get threshold for given slice.

   return fSliceInfos[slice].fThreshold;
}

//______________________________________________________________________________
void TEveCaloData::SetSliceColor(Int_t slice, Color_t col)
{
   // Set color for given slice.

   fSliceInfos[slice].fColor = col;
   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->AddStamp(TEveElement::kCBObjProps);
   }
}

//______________________________________________________________________________
Color_t TEveCaloData::GetSliceColor(Int_t slice) const
{
   // Get color for given slice.

   return fSliceInfos[slice].fColor;
}

//______________________________________________________________________________
void TEveCaloData::InvalidateUsersCellIdCache()
{
   // Invalidate cell ids cache on back ptr references.

   TEveCaloViz* calo;
   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      calo = dynamic_cast<TEveCaloViz*>(*i);
      calo->InvalidateCellIdCache();
      calo->StampObjProps();
   }
}

//______________________________________________________________________________
void TEveCaloData::DataChanged()
{
   // Tell users (TEveCaloViz instances using this data) that data
   // has changed and they should update the limits/scales etc.
   // This is done by calling TEveCaloViz::DataChanged().

   TEveCaloViz* calo;
   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      calo = dynamic_cast<TEveCaloViz*>(*i);
      calo->DataChanged();
      calo->StampObjProps();
   }
}

//______________________________________________________________________________
Float_t TEveCaloData::EtaToTheta(Float_t eta)
{
   using namespace TMath;

   if (eta < 0)
      return Pi() - 2*ATan(Exp(- Abs(eta)));
   else
      return 2*ATan(Exp(- Abs(eta)));
}


//==============================================================================
// TEveCaloDataVec
//==============================================================================

//______________________________________________________________________________
//
// Calo data for universal cell geometry.

ClassImp(TEveCaloDataVec);

//______________________________________________________________________________
TEveCaloDataVec::TEveCaloDataVec(Int_t nslices):
   TEveCaloData(),

   fTower(0),
   fEtaMin( 1e3),
   fEtaMax(-1e3),
   fPhiMin( 1e3),
   fPhiMax(-1e3)
{
   // Constructor.

   fSliceInfos.assign(nslices, SliceInfo_t());

   fSliceVec.assign(nslices, std::vector<Float_t> ());
}

//______________________________________________________________________________
TEveCaloDataVec::~TEveCaloDataVec()
{
   // Destructor.

   if (fEtaAxis) delete fEtaAxis;
   if (fPhiAxis) delete fPhiAxis;
}

//______________________________________________________________________________
Int_t TEveCaloDataVec::AddTower(Float_t etaMin, Float_t etaMax, Float_t phiMin, Float_t phiMax)
{
   // Add tower within eta/phi range.

   assert (etaMin < etaMax);
   assert (phiMin < phiMax);

   fGeomVec.push_back(CellGeom_t(etaMin, etaMax, phiMin, phiMax));

   for (vvFloat_i it=fSliceVec.begin(); it!=fSliceVec.end(); ++it)
      (*it).push_back(0);

   if (etaMin < fEtaMin) fEtaMin = etaMin;
   if (etaMax > fEtaMax) fEtaMax = etaMax;

   if (phiMin < fPhiMin) fPhiMin = phiMin;
   if (phiMax > fPhiMax) fPhiMax = phiMax;

   fTower = fGeomVec.size() - 1;
   return fTower;
}

//______________________________________________________________________________
void TEveCaloDataVec::FillSlice(Int_t slice, Float_t val)
{
   // Fill given slice in the current tower.

   fSliceVec[slice][fTower] = val;
}

//______________________________________________________________________________
void TEveCaloDataVec::FillSlice(Int_t slice, Int_t tower, Float_t val)
{
   // Fill given slice in a given tower.

   fSliceVec[slice][tower] = val;
}


//______________________________________________________________________________
void TEveCaloDataVec::GetCellList(Float_t eta, Float_t etaD,
                                  Float_t phi, Float_t phiD,
                                  TEveCaloData::vCellId_t &out) const
{
   // Get list of cell-ids for given eta/phi range.

   using namespace TMath;

   Float_t etaMin = eta - etaD*0.5;
   Float_t etaMax = eta + etaD*0.5;

   Float_t phiMin = phi - phiD*0.5;
   Float_t phiMax = phi + phiD*0.5;

   Int_t nS = fSliceVec.size();

   Int_t tower = 0;
   Float_t fracx=0, fracy=0, frac;
   Float_t minQ, maxQ;

   for(vCellGeom_ci i=fGeomVec.begin(); i!=fGeomVec.end(); i++)
   {
      const CellGeom_t &cg = *i;
      fracx = TEveUtil::GetFraction(etaMin, etaMax, cg.fEtaMin, cg.fEtaMax);
      if (fracx > 1e-3)
      {
         minQ = cg.fPhiMin;
         maxQ = cg.fPhiMax;

         if (maxQ < phiMin)
         {
            minQ += TwoPi(); maxQ += TwoPi();
         }
         else if (minQ > phiMax)
         {
            minQ -= TwoPi(); maxQ -= TwoPi();
         }

         if (maxQ >= phiMin && minQ <= phiMax)
         {
            fracy = TEveUtil::GetFraction(phiMin, phiMax, minQ, maxQ);
            if (fracy > 1e-3)
            {
               frac = fracx*fracy;
               for (Int_t s=0; s<nS; s++)
               {
                  if (fSliceVec[s][tower] > fSliceInfos[s].fThreshold)
                     out.push_back(CellId_t(tower, s, frac));
               }
            }
         }
      }
      tower++;
   }
}

//______________________________________________________________________________
void TEveCaloDataVec::Rebin(TAxis* ax, TAxis* ay, vCellId_t &ids, Bool_t et, RebinData_t& rdata) const
{
   // Rebin cells.

   rdata.fNSlices = GetNSlices();
   rdata.fBinData.assign((ax->GetNbins()+2)*(ay->GetNbins()+2), -1);

   CellData_t cd;
   Float_t left, right, up, down; // cell corners
   for (vCellId_i it = ids.begin(); it != ids.end(); ++it)
   {
      GetCellData(*it, cd);
      Int_t iMin = ax->FindBin(cd.EtaMin());
      Int_t iMax = ax->FindBin(cd.EtaMax());
      Int_t jMin = ay->FindBin(cd.PhiMin());
      Int_t jMax = ay->FindBin(cd.PhiMax());
      for (Int_t i = iMin; i <= iMax; ++i)
      {
         if (i < 0 || i > ax->GetNbins()) continue;
         left  = (i == iMin) ? cd.EtaMin() : ax->GetBinLowEdge(i);
         right = (i == iMax) ? cd.EtaMax() : ax->GetBinUpEdge(i);

         for (Int_t j = jMin; j <= jMax; ++j)
         {
            if (j < 0 || j > ay->GetNbins()) continue;
            down = (j == jMin) ? cd.PhiMin() : ay->GetBinLowEdge(j);
            up   = (j == jMax) ? cd.PhiMax() : ay->GetBinUpEdge(j);

            Float_t ratio = ((right-left)*(up-down))/(ax->GetBinWidth(i)*ay->GetBinWidth(j));
            if (ratio > 1e-6)
            {
               Float_t* slices = rdata.GetSliceVals(i+j*(ax->GetNbins()+2));
               slices[(*it).fSlice] += ratio* cd.Value(et);
            }
         }
      }
   }
}

//______________________________________________________________________________
void TEveCaloDataVec::GetCellData(const TEveCaloData::CellId_t &id,
                                  TEveCaloData::CellData_t& cellData) const
{
   // Get cell geometry and value from cell ID.

   cellData.CellGeom_t::operator=( fGeomVec[id.fTower] );
   cellData.fValue = fSliceVec[id.fSlice][id.fTower]*id.fFraction;
}

//______________________________________________________________________________
void TEveCaloDataVec::DataChanged()
{
   // Update limits and notify data users.

   using namespace TMath;

   // update max E/Et values

   fMaxValE = 0;
   fMaxValEt = 0;
   Float_t sum=0, cos=0;
   //   printf("geom vec %d slices %d\n",fGeomVec.size(), fSliceVec.size() );

   for (UInt_t tw=0; tw<fGeomVec.size(); tw++)
   {
      sum=0;
      for (vvFloat_i it=fSliceVec.begin(); it!=fSliceVec.end(); ++it)
         sum += (*it)[tw];

      if (sum > fMaxValEt ) fMaxValEt=sum;

      cos = Cos(2*ATan(Exp( -Abs(fGeomVec[tw].Eta()))));
      sum /= Abs(cos);
      if (sum > fMaxValE) fMaxValE=sum;
   }

   TEveCaloData::DataChanged();
}


//______________________________________________________________________________
void  TEveCaloDataVec::SetAxisFromBins(Double_t epsX, Double_t epsY)
{
   // Set XY axis from cells geometry.

   std::vector<Double_t> binX;
   std::vector<Double_t> binY;

   for(vCellGeom_ci i=fGeomVec.begin(); i!=fGeomVec.end(); i++)
   {
      const CellGeom_t &ch = *i;

      binX.push_back(ch.EtaMin());
      binX.push_back(ch.EtaMax());
      binY.push_back(ch.PhiMin());
      binY.push_back(ch.PhiMax());
   }

   std::sort(binX.begin(), binX.end());
   std::sort(binY.begin(), binY.end());

   Int_t cnt = 0;
   Double_t sum = 0;
   Double_t val;

   // X axis
   Double_t dx = binX.back() - binX.front();
   epsX *= dx;
   std::vector<Double_t> newX;
   newX.push_back(binX.front()); // underflow
   Int_t nX = binX.size()-1;
   for(Int_t i=0; i<nX; i++)
   {
      val = (sum +binX[i])/(cnt+1);
      if (binX[i+1] -val > epsX)
      {
         newX.push_back(val);
         cnt = 0;
         sum = 0;
      }
      else
      {
         sum += binX[i];
         cnt++;
      }
   }
   newX.push_back(binX.back()); // overflow

   // Y axis
   cnt = 0;
   sum = 0;
   std::vector<Double_t> newY;
   Double_t dy = binY.back() - binY.front();
   epsY *= dy;
   newY.push_back(binY.front());// underflow
   Int_t nY = binY.size()-1;
   for(Int_t i=0 ; i<nY; i++)
   {
      val = (sum +binY[i])/(cnt+1);
      if (binY[i+1] -val > epsY )
      {
         newY.push_back(val);
         cnt = 0;
         sum = 0;
      }
      else
      {
         sum += binY[i];
         cnt++;
      }

   }
   newY.push_back(binY.back()); // overflow

   if (fEtaAxis) delete fEtaAxis;
   if (fPhiAxis) delete fPhiAxis;

   fEtaAxis = new TAxis(newX.size()-1, &newX[0]);
   fPhiAxis = new TAxis(newY.size()-1, &newY[0]);
   fEtaAxis->SetNdivisions(510);
   fPhiAxis->SetNdivisions(510);
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

//______________________________________________________________________________
TEveCaloDataHist::TEveCaloDataHist():
   TEveCaloData(),

   fHStack(0)
{
   // Constructor.

   fHStack = new THStack();
   fEps    = 1e-5;
}

//______________________________________________________________________________
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
   fMaxValE  = 0;
   fMaxValEt = 0;

   if (GetNSlices() < 1) return;

   TH2* hist = GetHist(0);
   fEtaAxis  = hist->GetXaxis();
   fPhiAxis  = hist->GetYaxis();
   for (Int_t ieta = 1; ieta <= fEtaAxis->GetNbins(); ++ieta)
   {
      Double_t eta = fEtaAxis->GetBinCenter(ieta); // conversion E/Et
      for (Int_t iphi = 1; iphi <= fPhiAxis->GetNbins(); ++iphi)
      {
         Double_t value = 0;
         for (Int_t i = 0; i < GetNSlices(); ++i)
         {
            hist = GetHist(i);
            Int_t bin = hist->GetBin(ieta, iphi);
            value += hist->GetBinContent(bin);
         }

         if (value > fMaxValEt ) fMaxValEt = value;

         Double_t cos = Cos(2*ATan(Exp(-Abs(eta))));
         value /= Abs(cos);
         if (value > fMaxValE) fMaxValE = value;
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

   Float_t etaMin = eta - etaD*0.5 -fEps;
   Float_t etaMax = eta + etaD*0.5 +fEps;

   Float_t phiMin = phi - phiD*0.5 -fEps;
   Float_t phiMax = phi + phiD*0.5 +fEps;

   Int_t nEta = fEtaAxis->GetNbins();
   Int_t nPhi = fPhiAxis->GetNbins();
   Int_t nSlices = GetNSlices();

   TH2F* hist = GetHist(0);
   Int_t bin  = 0;

   Bool_t accept;
   for (Int_t ieta = 1; ieta <= nEta; ++ieta)
   {
      if (fEtaAxis->GetBinLowEdge(ieta) >= etaMin && fEtaAxis->GetBinUpEdge(ieta) <= etaMax)
      {
         for (Int_t iphi = 1; iphi <= nPhi; ++iphi)
         {
            if (fWrapTwoPi )
            {
               accept = TEveUtil::IsU1IntervalContainedByMinMax
                  (phiMin, phiMax, fPhiAxis->GetBinLowEdge(iphi), fPhiAxis->GetBinUpEdge(iphi));
            }
            else
            {
               accept = fPhiAxis->GetBinLowEdge(iphi) >= phiMin &&  fPhiAxis->GetBinUpEdge(iphi) <= phiMax &&
                  fPhiAxis->GetBinLowEdge(iphi) >= phiMin &&  fPhiAxis->GetBinUpEdge(iphi) <= phiMax;
            }

            if (accept)
            {
               for (Int_t s = 0; s < nSlices; ++s)
               {
                  hist = GetHist(s);
                  bin = hist->GetBin(ieta, iphi);
                  if (hist->GetBinContent(bin) > fSliceInfos[s].fThreshold)
                     out.push_back(TEveCaloData::CellId_t(bin, s));
               } // hist slices
            }
         } // phi bins
      }
   } // eta bins
}


//______________________________________________________________________________
void TEveCaloDataHist::Rebin(TAxis* ax, TAxis* ay, TEveCaloData::vCellId_t &ids, Bool_t et, RebinData_t &rdata) const
{
   rdata.fNSlices = GetNSlices();
   rdata.fBinData.assign((ax->GetNbins()+2)*(ay->GetNbins()+2), -1);
   TEveCaloData::CellData_t cd;
   Float_t *val;
   Int_t i, j, w;
   Int_t binx, biny;
   Int_t bin;

   for (vCellId_i it=ids.begin(); it!=ids.end(); ++it)
   {
      GetCellData(*it, cd);
      GetHist(it->fSlice)->GetBinXYZ((*it).fTower, i, j, w);
      binx = ax->FindBin(fEtaAxis->GetBinCenter(i));
      biny = ay->FindBin(fPhiAxis->GetBinCenter(j));
      bin = biny*(ax->GetNbins()+2)+binx;
      val = rdata.GetSliceVals(bin);
      Double_t ratio = (fEtaAxis->GetBinWidth(i)*fPhiAxis->GetBinWidth(j))/(ax->GetBinWidth(binx)*ay->GetBinWidth(biny));
      val[(*it).fSlice] += cd.Value(et)*ratio;
   }
}

//______________________________________________________________________________
void TEveCaloDataHist::GetCellData(const TEveCaloData::CellId_t &id,
                                   TEveCaloData::CellData_t& cellData) const
{
   // Get cell geometry and value from cell ID.

   TH2F* hist = GetHist(id.fSlice);

   Int_t x, y, z;
   hist->GetBinXYZ(id.fTower, x, y, z);

   cellData.fValue =  hist->GetBinContent(id.fTower);
   cellData.Configure(hist->GetXaxis()->GetBinLowEdge(x),
                      hist->GetXaxis()->GetBinUpEdge(x),
                      hist->GetYaxis()->GetBinLowEdge(y),
                      hist->GetYaxis()->GetBinUpEdge(y));
}


//______________________________________________________________________________
Int_t TEveCaloDataHist::AddHistogram(TH2F* hist)
{
   // Add new slice to calo tower. Updates cached variables fMaxValE
   // and fMaxValEt
   // Return last index in the vector of slice infos.

   fHStack->Add(hist);
   fSliceInfos.push_back(SliceInfo_t());
   fSliceInfos.back().fName  = hist->GetName();
   fSliceInfos.back().fColor = hist->GetLineColor();
   
   DataChanged();
   
   return fSliceInfos.size() - 1;
}

//______________________________________________________________________________
TH2F* TEveCaloDataHist::GetHist(Int_t slice) const
{
   // Get histogram in given slice.
   
   return (TH2F*) fHStack->GetHists()->At(slice);
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
