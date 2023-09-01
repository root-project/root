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

#include "TGLSelectRecord.h"

#include "TAxis.h"
#include "THStack.h"
#include "TH2.h"
#include "TMath.h"
#include "TList.h"

#include <cassert>
#include <algorithm>
#include <set>

/** \class  TEveCaloData::CellGeom_t
\ingroup TEve
Cell geometry inner structure.
*/

////////////////////////////////////////////////////////////////////////////////
/// Print member data.

void TEveCaloData::CellGeom_t::Dump() const
{
   printf("%f, %f %f, %f \n", fEtaMin, fEtaMax, fPhiMin, fPhiMax);
}

////////////////////////////////////////////////////////////////////////////////

void TEveCaloData::CellGeom_t::Configure(Float_t etaMin, Float_t etaMax, Float_t phiMin, Float_t phiMax)
{
   fEtaMin = etaMin;
   fEtaMax = etaMax;

   fPhiMin = phiMin;
   fPhiMax = phiMax;

   // Complain if phi is out of [-2*pi, 2*pi] range.
   if (fPhiMin < - TMath::TwoPi() || fPhiMin > TMath::TwoPi() ||
       fPhiMax < - TMath::TwoPi() || fPhiMax > TMath::TwoPi())
   {
     ::Error("TEveCaloData::CellGeom_t::Configure", "phiMin and phiMax should be between -2*pi and 2*pi (min=%f, max=%f). RhoZ projection will be wrong.",
             fPhiMin, fPhiMax);
   }

   fThetaMin = EtaToTheta(fEtaMax);
   fThetaMax = EtaToTheta(fEtaMin);
}

/** \class TEveCaloData::CellData_t
\ingroup TEve
Cell data inner structure.
*/

////////////////////////////////////////////////////////////////////////////////
/// Return energy value associated with the cell, usually Et.
/// If isEt is false it is transformed into energy E.

Float_t TEveCaloData::CellData_t::Value(Bool_t isEt) const
{
   if (isEt)
      return fValue;
   else
      return TMath::Abs(fValue/TMath::Sin(Theta()));
}

////////////////////////////////////////////////////////////////////////////////
/// Print member data.

void TEveCaloData::CellData_t::Dump() const
{
   printf("%f, %f %f, %f \n", fEtaMin, fEtaMax, fPhiMin, fPhiMax);
}

////////////////////////////////////////////////////////////////////////////////

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

/** \class TEveCaloData
\ingroup TEve
A central manager for calorimeter event data. It provides a list of
cells within requested phi and eta range.
*/

ClassImp(TEveCaloData);

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
/// Virtual method TEveElement::UnSelect.
/// Clear selected towers when deselected.

void TEveCaloData::UnSelected()
{
   fCellsSelected.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual method TEveElement::UnHighlighted.

void TEveCaloData::UnHighlighted()
{
   fCellsHighlighted.clear();
}

////////////////////////////////////////////////////////////////////////////////

TString TEveCaloData::GetHighlightTooltip()
{
   if (fCellsHighlighted.empty()) return "";

   CellData_t cellData;

   Bool_t single = fCellsHighlighted.size() == 1;
   Float_t sum = 0;
   TString s;
   for (vCellId_i i = fCellsHighlighted.begin(); i!=fCellsHighlighted.end(); ++i)
   {
      GetCellData(*i, cellData);

      s += TString::Format("%s %.2f (%.3f, %.3f)",
                           fSliceInfos[i->fSlice].fName.Data(), cellData.fValue,
                           cellData.Eta(), cellData.Phi());

      if (single) return s;
      s += "\n";
      sum += cellData.fValue;
   }
   s += TString::Format("Sum = %.2f", sum);
   return s;
}

////////////////////////////////////////////////////////////////////////////////
/// Populate set impSelSet with derived / dependant elements.
///

void TEveCaloData::FillImpliedSelectedSet(Set_t& impSelSet)
{
   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      impSelSet.insert(*i);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Print selected cells info.

void TEveCaloData::PrintCellsSelected()
{
   printf("%d Selected selected cells:\n", (Int_t)fCellsSelected.size());
   CellData_t cellData;

   for (vCellId_i i = fCellsSelected.begin(); i != fCellsSelected.end(); ++i)
   {
      GetCellData(*i, cellData);
      printf("Tower [%d] Slice [%d] Value [%.2f] ", i->fTower, i->fSlice, cellData.fValue);
      printf("Eta:(%f, %f) Phi(%f, %f)\n",  cellData.fEtaMin, cellData.fEtaMax, cellData.fPhiMin, cellData.fPhiMax);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Process newly selected cells with given select-record.
/// Secondary-select status is set.
/// CellSelectionChanged() is called if needed.

void TEveCaloData::ProcessSelection(vCellId_t& sel_cells, TGLSelectRecord& rec)
{
   typedef std::set<CellId_t>           sCellId_t;
   typedef std::set<CellId_t>::iterator sCellId_i;

   struct helper
   {
      static void fill_cell_set(sCellId_t& cset, vCellId_t& cvec)
      {
         for (vCellId_i i = cvec.begin(); i != cvec.end(); ++i)
            cset.insert(*i);
      }
      static void fill_cell_vec(vCellId_t& cvec, sCellId_t& cset)
      {
         for (sCellId_i i = cset.begin(); i != cset.end(); ++i)
            cvec.push_back(*i);
      }
   };

   vCellId_t& cells = rec.GetHighlight() ? fCellsHighlighted : fCellsSelected;

   if (cells.empty())
   {
      if (!sel_cells.empty())
      {
         cells.swap(sel_cells);
         rec.SetSecSelResult(TGLSelectRecord::kEnteringSelection);
      }
   }
   else
   {
      if (!sel_cells.empty())
      {
         if (rec.GetMultiple())
         {
            sCellId_t cs;
            helper::fill_cell_set(cs, cells);
            for (vCellId_i i = sel_cells.begin(); i != sel_cells.end(); ++i)
            {
               std::set<CellId_t>::iterator csi = cs.find(*i);
               if (csi == cs.end())
                  cs.insert(*i);
               else
                  cs.erase(csi);
            }
            cells.clear();
            if (cs.empty())
            {
               rec.SetSecSelResult(TGLSelectRecord::kLeavingSelection);
            }
            else
            {
               helper::fill_cell_vec(cells, cs);
               rec.SetSecSelResult(TGLSelectRecord::kModifyingInternalSelection);
            }
         }
         else
         {
            Bool_t differ = kFALSE;
            if (cells.size() == sel_cells.size())
            {
               sCellId_t cs;
               helper::fill_cell_set(cs, cells);
               for (vCellId_i i = sel_cells.begin(); i != sel_cells.end(); ++i)
               {
                  if (cs.find(*i) == cs.end())
                  {
                     differ = kTRUE;
                     break;
                  }
               }
            }
            else
            {
               differ = kTRUE;
            }
            if (differ)
            {
               cells.swap(sel_cells);
               rec.SetSecSelResult(TGLSelectRecord::kModifyingInternalSelection);
            }
         }
      }
      else
      {
         if (!rec.GetMultiple())
         {
            cells.clear();
            rec.SetSecSelResult(TGLSelectRecord::kLeavingSelection);
         }
      }
   }

   if (rec.GetSecSelResult() != TGLSelectRecord::kNone)
   {
      CellSelectionChanged();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set threshold for given slice.

void TEveCaloData::SetSliceThreshold(Int_t slice, Float_t val)
{
   fSliceInfos[slice].fThreshold = val;
   InvalidateUsersCellIdCache();
}

////////////////////////////////////////////////////////////////////////////////
/// Get threshold for given slice.

Float_t TEveCaloData::GetSliceThreshold(Int_t slice) const
{
   return fSliceInfos[slice].fThreshold;
}

////////////////////////////////////////////////////////////////////////////////
/// Set color for given slice.

void TEveCaloData::SetSliceColor(Int_t slice, Color_t col)
{
   fSliceInfos[slice].fColor = col;
   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->AddStamp(TEveElement::kCBObjProps);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get color for given slice.

Color_t TEveCaloData::GetSliceColor(Int_t slice) const
{
   return fSliceInfos[slice].fColor;
}

////////////////////////////////////////////////////////////////////////////////
/// Set transparency for given slice.

void TEveCaloData::SetSliceTransparency(Int_t slice, Char_t t)
{
   fSliceInfos[slice].fTransparency = t;
   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->AddStamp(TEveElement::kCBObjProps);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get transparency for given slice.

Char_t TEveCaloData::GetSliceTransparency(Int_t slice) const
{
   return fSliceInfos[slice].fTransparency;
}

////////////////////////////////////////////////////////////////////////////////
/// Invalidate cell ids cache on back ptr references.

void TEveCaloData::InvalidateUsersCellIdCache()
{
   TEveCaloViz* calo;
   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      calo = dynamic_cast<TEveCaloViz*>(*i);
      calo->InvalidateCellIdCache();
      calo->StampObjProps();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Tell users (TEveCaloViz instances using this data) that data
/// has changed and they should update the limits/scales etc.
/// This is done by calling TEveCaloViz::DataChanged().

void TEveCaloData::DataChanged()
{
   TEveCaloViz* calo;
   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      calo = dynamic_cast<TEveCaloViz*>(*i);
      calo->DataChanged();
      calo->StampObjProps();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Tell users (TEveCaloViz instances using this data) that cell selection
/// has changed and they should update selection cache if necessary.
/// This is done by calling TEveCaloViz::CellSelectionChanged().

void TEveCaloData::CellSelectionChanged()
{
   TEveCaloViz* calo;
   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      calo = dynamic_cast<TEveCaloViz*>(*i);
      calo->CellSelectionChanged();
      calo->StampColorSelection();
   }
}

////////////////////////////////////////////////////////////////////////////////

Float_t TEveCaloData::EtaToTheta(Float_t eta)
{
   using namespace TMath;

   if (eta < 0)
      return Pi() - 2*ATan(Exp(- Abs(eta)));
   else
      return 2*ATan(Exp(- Abs(eta)));
}


/** \class  TEveCaloDataVec
\ingroup TEve
Calo data for universal cell geometry.
*/

ClassImp(TEveCaloDataVec);

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveCaloDataVec::~TEveCaloDataVec()
{
   if (fEtaAxis) delete fEtaAxis;
   if (fPhiAxis) delete fPhiAxis;
}

////////////////////////////////////////////////////////////////////////////////
/// Add new slice.

Int_t TEveCaloDataVec::AddSlice()
{
  fSliceInfos.push_back(SliceInfo_t());
  fSliceVec.push_back(std::vector<Float_t> ());
  fSliceVec.back().resize(fGeomVec.size(), 0.f);

  return fSliceInfos.size() - 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Add tower within eta/phi range.

Int_t TEveCaloDataVec::AddTower(Float_t etaMin, Float_t etaMax, Float_t phiMin, Float_t phiMax)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Fill given slice in the current tower.

void TEveCaloDataVec::FillSlice(Int_t slice, Float_t val)
{
   fSliceVec[slice][fTower] = val;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill given slice in a given tower.

void TEveCaloDataVec::FillSlice(Int_t slice, Int_t tower, Float_t val)
{
   fSliceVec[slice][tower] = val;
}


////////////////////////////////////////////////////////////////////////////////
/// Get list of cell-ids for given eta/phi range.

void TEveCaloDataVec::GetCellList(Float_t eta, Float_t etaD,
                                  Float_t phi, Float_t phiD,
                                  TEveCaloData::vCellId_t &out) const
{
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

         if (fWrapTwoPi)
         {
            if (maxQ < phiMin)
            {
               minQ += TwoPi(); maxQ += TwoPi();
            }
            else if (minQ > phiMax)
            {
               minQ -= TwoPi(); maxQ -= TwoPi();
            }
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

////////////////////////////////////////////////////////////////////////////////
/// Rebin cells.

void TEveCaloDataVec::Rebin(TAxis* ax, TAxis* ay, vCellId_t &ids, Bool_t et, RebinData_t& rdata) const
{
   rdata.fNSlices = GetNSlices();
   rdata.fBinData.assign((ax->GetNbins()+2)*(ay->GetNbins()+2), -1);

   CellData_t cd;
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
         for (Int_t j = jMin; j <= jMax; ++j)
         {
            if (j < 0 || j > ay->GetNbins()) continue;

            Double_t ratio = TEveUtil::GetFraction(ax->GetBinLowEdge(i), ax->GetBinUpEdge(i), cd.EtaMin(), cd.EtaMax())
                           * TEveUtil::GetFraction(ay->GetBinLowEdge(j), ay->GetBinUpEdge(j), cd.PhiMin(), cd.PhiMax());

            if (ratio > 1e-6f)
            {
               Float_t* slices = rdata.GetSliceVals(i + j*(ax->GetNbins()+2));
               slices[(*it).fSlice] += ratio * cd.Value(et);
            }
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get cell geometry and value from cell ID.

void TEveCaloDataVec::GetCellData(const TEveCaloData::CellId_t &id,
                                  TEveCaloData::CellData_t& cellData) const
{
   cellData.CellGeom_t::operator=( fGeomVec[id.fTower] );
   cellData.fValue = fSliceVec[id.fSlice][id.fTower];
}

////////////////////////////////////////////////////////////////////////////////
/// Update limits and notify data users.

void TEveCaloDataVec::DataChanged()
{
   using namespace TMath;

   // update max E/Et values

   fMaxValE = 0;
   fMaxValEt = 0;
   Float_t sum=0;
   //   printf("geom vec %d slices %d\n",fGeomVec.size(), fSliceVec.size() );

   for (UInt_t tw=0; tw<fGeomVec.size(); tw++)
   {
      sum=0;
      for (vvFloat_i it=fSliceVec.begin(); it!=fSliceVec.end(); ++it)
         sum += (*it)[tw];

      if (sum > fMaxValEt ) fMaxValEt=sum;

      sum /= Abs(Sin(EtaToTheta(fGeomVec[tw].Eta())));

      if (sum > fMaxValE) fMaxValE=sum;
   }

   TEveCaloData::DataChanged();
}


////////////////////////////////////////////////////////////////////////////////
/// Set XY axis from cells geometry.

void  TEveCaloDataVec::SetAxisFromBins(Double_t epsX, Double_t epsY)
{
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

/** \class TEveCaloDataHist
\ingroup TEve
A central manager for calorimeter data of an event written in TH2F.
X axis is used for eta and Y axis for phi.
*/

ClassImp(TEveCaloDataHist);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveCaloDataHist::TEveCaloDataHist():
   TEveCaloData(),

   fHStack(0)
{
   fHStack = new THStack();
   fEps    = 1e-5;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveCaloDataHist::~TEveCaloDataHist()
{
   delete fHStack;
}

////////////////////////////////////////////////////////////////////////////////
/// Update limits and notify data users.

void TEveCaloDataHist::DataChanged()
{
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

         value /= Abs(Sin(EtaToTheta(eta)));

         if (value > fMaxValE) fMaxValE = value;
      }
   }
   TEveCaloData::DataChanged();
}

////////////////////////////////////////////////////////////////////////////////
/// Get list of cell IDs in given eta and phi range.

void TEveCaloDataHist::GetCellList(Float_t eta, Float_t etaD,
                                   Float_t phi, Float_t phiD,
                                   TEveCaloData::vCellId_t &out) const
{
   using namespace TMath;

   Float_t etaMin = eta - etaD*0.5 -fEps;
   Float_t etaMax = eta + etaD*0.5 +fEps;

   Float_t phiMin = phi - phiD*0.5 -fEps;
   Float_t phiMax = phi + phiD*0.5 +fEps;

   Int_t nEta = fEtaAxis->GetNbins();
   Int_t nPhi = fPhiAxis->GetNbins();
   Int_t nSlices = GetNSlices();

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
                  TH2F *hist = GetHist(s);
                  bin = hist->GetBin(ieta, iphi);
                  if (hist->GetBinContent(bin) > fSliceInfos[s].fThreshold)
                     out.push_back(TEveCaloData::CellId_t(bin, s));
               } // hist slices
            }
         } // phi bins
      }
   } // eta bins
}

////////////////////////////////////////////////////////////////////////////////
/// Rebin

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
      Double_t ratio = TEveUtil::GetFraction(ax->GetBinLowEdge(binx), ax->GetBinUpEdge(binx), cd.EtaMin(), cd.EtaMax())
                     * TEveUtil::GetFraction(ay->GetBinLowEdge(biny), ay->GetBinUpEdge(biny), cd.PhiMin(), cd.PhiMax());

      val[(*it).fSlice] += cd.Value(et)*ratio;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get cell geometry and value from cell ID.

void TEveCaloDataHist::GetCellData(const TEveCaloData::CellId_t &id,
                                   TEveCaloData::CellData_t& cellData) const
{
   TH2F* hist = GetHist(id.fSlice);

   Int_t x, y, z;
   hist->GetBinXYZ(id.fTower, x, y, z);

   cellData.fValue =  hist->GetBinContent(id.fTower);
   cellData.Configure(hist->GetXaxis()->GetBinLowEdge(x),
                      hist->GetXaxis()->GetBinUpEdge(x),
                      hist->GetYaxis()->GetBinLowEdge(y),
                      hist->GetYaxis()->GetBinUpEdge(y));
}

////////////////////////////////////////////////////////////////////////////////
/// Add new slice to calo tower. Updates cached variables fMaxValE
/// and fMaxValEt
/// Return last index in the vector of slice infos.

Int_t TEveCaloDataHist::AddHistogram(TH2F* hist)
{
   fHStack->Add(hist);
   fSliceInfos.push_back(SliceInfo_t());
   fSliceInfos.back().fName  = hist->GetName();
   fSliceInfos.back().fColor = hist->GetLineColor();

   DataChanged();

   return fSliceInfos.size() - 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Get histogram in given slice.

TH2F* TEveCaloDataHist::GetHist(Int_t slice) const
{
   assert(slice >= 0 && slice < fHStack->GetHists()->GetSize());
   return (TH2F*) fHStack->GetHists()->At(slice);
}

////////////////////////////////////////////////////////////////////////////////
/// Get eta limits.

void TEveCaloDataHist::GetEtaLimits(Double_t &min, Double_t &max) const
{
   min = fEtaAxis->GetXmin();
   max = fEtaAxis->GetXmax();
}

////////////////////////////////////////////////////////////////////////////////
/// Get phi limits.

void TEveCaloDataHist::GetPhiLimits(Double_t &min, Double_t &max) const
{
   min = fPhiAxis->GetXmin();
   max = fPhiAxis->GetXmax();
}
