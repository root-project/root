// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TEveProjections.h"
#include "TEveProjectionManager.h"
#include "TEveRGBAPalette.h"
#include "TEveText.h"
#include "TEveTrans.h"

#include "TClass.h"
#include "TMathBase.h"
#include "TMath.h"
#include "TAxis.h"

#include "TGLUtil.h"

#include <cassert>

//==============================================================================
// TEveCaloViz
//==============================================================================

//______________________________________________________________________________
//
// Base class for calorimeter data visualization.
// See TEveCalo2D and TEveCalo3D for concrete implementations.

ClassImp(TEveCaloViz);

//______________________________________________________________________________
TEveCaloViz::TEveCaloViz(TEveCaloData* data, const char* n, const char* t) :
   TEveElement(),
   TNamed(n, t),
   TEveProjectable(),

   fData(0),
   fCellIdCacheOK(kFALSE),

   fEtaMin(-10),
   fEtaMax(10),

   fPhi(0.),
   fPhiOffset(TMath::Pi()),

   fAutoRange(kTRUE),

   fBarrelRadius(-1.f),
   fEndCapPosF(-1.f),
   fEndCapPosB(-1.f),

   fPlotEt(kTRUE),

   fMaxTowerH(100),
   fScaleAbs(kFALSE),
   fMaxValAbs(100),

   fValueIsColor(kFALSE),
   fPalette(0)
{
   // Constructor.

   fPickable = kTRUE;
   SetElementNameTitle(n, t);
   SetData(data);
}

//______________________________________________________________________________
TEveCaloViz::~TEveCaloViz()
{
   // Destructor.

   if (fPalette) fPalette->DecRefCount();
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetDataSliceThreshold(Int_t slice) const
{
   // Get threshold for given slice.

   return fData->RefSliceInfo(slice).fThreshold;
}

//______________________________________________________________________________
TEveElement* TEveCaloViz::ForwardSelection()
{
   // Management of selection state and ownership of selected cell list
   // is done in TEveCaloData. This is a reason selection is forwared to it.

   return fData;
}

//______________________________________________________________________________
TEveElement* TEveCaloViz::ForwardEdit()
{
   // Management of selection state and ownership of selected cell list
   // is done in TEveCaloData. We still want GUI editor to disply
   // concrete calo-viz object.

   return this;
}

//______________________________________________________________________________
void TEveCaloViz::SetDataSliceThreshold(Int_t slice, Float_t val)
{
   // Set threshold for given slice.

   fData->SetSliceThreshold(slice, val);
}

//______________________________________________________________________________
Color_t TEveCaloViz::GetDataSliceColor(Int_t slice) const
{
   // Get slice color from data.

   return fData->RefSliceInfo(slice).fColor;
}

//______________________________________________________________________________
void TEveCaloViz::SetDataSliceColor(Int_t slice, Color_t col)
{
   // Set slice color in data.

   fData->SetSliceColor(slice, col);
}

//______________________________________________________________________________
void TEveCaloViz::SetEta(Float_t l, Float_t u)
{
   // Set eta range.

   fEtaMin=l;
   fEtaMax=u;

   InvalidateCellIdCache();
}

//______________________________________________________________________________
void TEveCaloViz::SetPlotEt(Bool_t isEt)
{
   // Set E/Et plot.

   fPlotEt=isEt;
   if (fPalette)
      fPalette->SetLimits(0, TMath::CeilNint(GetMaxVal()));

   InvalidateCellIdCache();
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetMaxVal() const
{

   // Get maximum plotted value.

   return fData->GetMaxVal(fPlotEt);

}

//______________________________________________________________________________
void TEveCaloViz::SetPhiWithRng(Float_t phi, Float_t rng)
{
   // Set phi range.

   using namespace TMath;

   fPhi = phi;
   fPhiOffset = rng;

   InvalidateCellIdCache();
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetTransitionTheta() const
{
   // Get transition angle between barrel and end-cap cells, assuming fEndCapPosF = -fEndCapPosB.

   return TMath::ATan(fBarrelRadius/fEndCapPosF);
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetTransitionEta() const
{
   // Get transition eta between barrel and end-cap cells, assuming fEndCapPosF = -fEndCapPosB.

   using namespace TMath;
   Float_t t = GetTransitionTheta()*0.5f;
   return -Log(Tan(t));
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetTransitionThetaForward() const
{
   // Get transition angle between barrel and forward end-cap cells.

   return TMath::ATan(fBarrelRadius/fEndCapPosF);
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetTransitionEtaForward() const
{
   // Get transition eta between barrel and forward end-cap cells.

   using namespace TMath;
   Float_t t = GetTransitionThetaForward()*0.5f;
   return -Log(Tan(t));
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetTransitionThetaBackward() const
{
   // Get transition angle between barrel and backward end-cap cells.

   return TMath::ATan(fBarrelRadius/fEndCapPosB);
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetTransitionEtaBackward() const
{
   // Get transition eta between barrel and backward end-cap cells.

   using namespace TMath;
   Float_t t = GetTransitionThetaBackward()*0.5f;
   //negative theta means negative eta
   return Log(-Tan(t));
}


//______________________________________________________________________________
void TEveCaloViz::SetData(TEveCaloData* data)
{
   // Set calorimeter event data.


   if (data == fData) return;
   if (fData) fData->RemoveElement(this);
   fData = data;
   if (fData)
   {
      fData->AddElement(this);
      DataChanged();
   }
}

//______________________________________________________________________________
void TEveCaloViz::DataChanged()
{
   // Update setting and cache on data changed.
   // Called from TEvecaloData::BroadcastDataChange()

   Double_t min, max, delta;

   fData->GetEtaLimits(min, max);
   if (fAutoRange) {
      fEtaMin = min;
      fEtaMax = max;
   } else {
      if (fEtaMin < min) fEtaMin = min;
      if (fEtaMax > max) fEtaMax = max;
   }

   fData->GetPhiLimits(min, max);
   delta = 0.5*(max - min);
   if (fAutoRange || fPhi < min || fPhi > max) {
      fPhi       = 0.5*(max + min);
      fPhiOffset = delta;
   } else {
      if (fPhiOffset > delta) fPhiOffset = delta;
   }

   if (fPalette)
   {
      Int_t hlimit = TMath::CeilNint(GetMaxVal());
      fPalette->SetLimits(0, hlimit);
      fPalette->SetMin(0);
      fPalette->SetMax(hlimit);
   }

   InvalidateCellIdCache();
}

//______________________________________________________________________________
Bool_t TEveCaloViz::AssertCellIdCache() const
{
   // Assert cell id cache is ok.
   // Returns true if the cache has been updated.
 
   TEveCaloViz* cv = const_cast<TEveCaloViz*>(this);
   if (!fCellIdCacheOK) {
      cv->BuildCellIdCache();
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TEveCaloViz::CellInEtaPhiRng(TEveCaloData::CellData_t& cellData) const
{
   // Returns true if given cell is in the ceta phi range.

   if (cellData.EtaMin() >= fEtaMin && cellData.EtaMax() <= fEtaMax)
   {
      if (TEveUtil::IsU1IntervalContainedByMinMax
          (fPhi-fPhiOffset, fPhi+fPhiOffset, cellData.PhiMin(), cellData.PhiMax()))
         return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveCaloViz::AssignCaloVizParameters(TEveCaloViz* m)
{
   // Assign paramteres from given model.

   SetData(m->fData);

   fEtaMin    = m->fEtaMin;
   fEtaMax    = m->fEtaMax;

   fPhi       = m->fPhi;
   fPhiOffset = m->fPhiOffset;

   fBarrelRadius = m->fBarrelRadius;
   fEndCapPosF    = m->fEndCapPosF;
   fEndCapPosB    = m->fEndCapPosB;

   if (m->fPalette)
   {
      TEveRGBAPalette& mp = * m->fPalette;
      if (fPalette) fPalette->DecRefCount();
      fPalette = new TEveRGBAPalette(mp.GetMinVal(), mp.GetMaxVal(), mp.GetInterpolate());
      fPalette->SetDefaultColor(mp.GetDefaultColor());
   }
}

//______________________________________________________________________________
void TEveCaloViz::SetPalette(TEveRGBAPalette* p)
{
   // Set TEveRGBAPalette object pointer.

   if ( fPalette == p) return;
   if (fPalette) fPalette->DecRefCount();
   fPalette = p;
   if (fPalette) fPalette->IncRefCount();
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetValToHeight() const
{
   // Get transformation factor from E/Et to height

   if (fScaleAbs)
   {
      return fMaxTowerH/fMaxValAbs;
   }
   else
   {
     if (fData->Empty())
       return 1;

      return fMaxTowerH/fData->GetMaxVal(fPlotEt);
   }
}

//______________________________________________________________________________
TEveRGBAPalette* TEveCaloViz::AssertPalette()
{
   // Make sure the TEveRGBAPalette pointer is not null.
   // If it is not set, a new one is instantiated and the range is set
   // to current min/max signal values.

   if (fPalette == 0) {
      fPalette = new TEveRGBAPalette;
      fPalette->SetDefaultColor((Color_t)4);

      Int_t hlimit = TMath::CeilNint(GetMaxVal());
      fPalette->SetLimits(0, hlimit);
      fPalette->SetMin(0);
      fPalette->SetMax(hlimit);

   }
   return fPalette;
}

//______________________________________________________________________________
void TEveCaloViz::Paint(Option_t* /*option*/)
{
   // Paint this object. Only direct rendering is supported.

   if (fData)
   {
      PaintStandard(this);
   }
}

//______________________________________________________________________________
TClass* TEveCaloViz::ProjectedClass(const TEveProjection*) const
{
   // Virtual from TEveProjectable, returns TEveCalo2D class.

   return TEveCalo2D::Class();
}

//______________________________________________________________________________
void TEveCaloViz::SetupColorHeight(Float_t value, Int_t slice, Float_t& outH) const
{
   // Set color and height for a given value and slice using slice color or TEveRGBAPalette.

   if (fValueIsColor)
   {
      outH = GetValToHeight()*fData->GetMaxVal(fPlotEt);
      UChar_t c[4];
      fPalette->ColorFromValue((Int_t)value, c);
      c[3] = fData->GetSliceTransparency(slice);
      TGLUtil::Color4ubv(c);
   }
   else
   {
      TGLUtil::ColorTransparency(fData->GetSliceColor(slice), fData->GetSliceTransparency(slice));
      outH = GetValToHeight()*value;
   }
}


//==============================================================================
// TEveCalo3D
//==============================================================================

//______________________________________________________________________________
//
// Visualization of a calorimeter event data in 3D.

ClassImp(TEveCalo3D);


TEveCalo3D::TEveCalo3D(TEveCaloData* d, const char* n, const char* t):
   TEveCaloViz(d, n, t),

   fRnrEndCapFrame    (kTRUE),
   fRnrBarrelFrame    (kTRUE),
   fFrameWidth        (0.5),
   fFrameColor        (kGray+1),
   fFrameTransparency (80)
{

   // Constructor.

   fCanEditMainColor        = kTRUE;
   fCanEditMainTransparency = kTRUE;
   fMainColorPtr = &fFrameColor;
}

//______________________________________________________________________________
void TEveCalo3D::BuildCellIdCache()
{
   // Build list of drawn cell IDs. See TEveCalo3DGL::DirectDraw().

   fCellList.clear();

   fData->GetCellList(GetEta(), GetEtaRng(), GetPhi(), GetPhiRng(), fCellList);
   fCellIdCacheOK = kTRUE;
}

//______________________________________________________________________________
void TEveCalo3D::ComputeBBox()
{
   // Fill bounding-box information of the base-class TAttBBox (virtual method).
   // If member 'TEveFrameBox* fFrame' is set, frame's corners are used as bbox.

   BBoxInit();

   Float_t th = (fData) ? GetValToHeight() * fData->GetMaxVal(fPlotEt) : 0;

   fBBox[0] = -fBarrelRadius - th;
   fBBox[1] =  fBarrelRadius + th;
   fBBox[2] =  fBBox[0];
   fBBox[3] =  fBBox[1];
   fBBox[4] =  fEndCapPosB - th;
   fBBox[5] =  fEndCapPosF + th;
}


//==============================================================================
// TEveCalo2D
//==============================================================================

//______________________________________________________________________________
//
// Visualization of a calorimeter event data in 2D.

ClassImp(TEveCalo2D);

//______________________________________________________________________________
TEveCalo2D::TEveCalo2D(const char* n, const char* t):
   TEveCaloViz(0, n, t),
   TEveProjected(),
   fOldProjectionType(TEveProjection::kPT_Unknown),
   fMaxESumBin( 0),
   fMaxEtSumBin(0)
{
   // Constructor.
}

//______________________________________________________________________________
TEveCalo2D::~TEveCalo2D()
{
   // Destructor.

   TEveCaloData::vCellId_t* cids;
   UInt_t n;

   // clear selected cell ids
   n = fCellListsSelected.size();
   for(UInt_t i = 0; i < n; ++i) {
      cids = fCellListsSelected[i];
      if (cids) {
         cids->clear(); delete cids;
      }
   }
   fCellListsSelected.clear();

   // clear all cell dds
   n = fCellLists.size();
   for(UInt_t i = 0; i < n; ++i) {
      cids = fCellLists[i];
      if (cids) {
         cids->clear(); delete cids;
      }
   }
   fCellLists.clear();
}

//______________________________________________________________________________
void TEveCalo2D::UpdateProjection()
{
   // This is virtual method from base-class TEveProjected.

   if (fManager->GetProjection()->GetType() != fOldProjectionType)
   {
      fCellIdCacheOK=kFALSE;
      fOldProjectionType = fManager->GetProjection()->GetType();
   }
   ComputeBBox();
}

//______________________________________________________________________________
void TEveCalo2D::SetProjection(TEveProjectionManager* mng, TEveProjectable* model)
{
   // Set projection manager and model object.

   TEveProjected::SetProjection(mng, model);
   TEveCaloViz* viz = dynamic_cast<TEveCaloViz*>(model);
   AssignCaloVizParameters(viz);
}

//______________________________________________________________________________
void TEveCalo2D::BuildCellIdCache()
{
   // Build lists of drawn cell IDs. See TEveCalo2DGL::DirecDraw().

   // clear old cache
   for (vBinCells_i it = fCellLists.begin(); it != fCellLists.end(); it++)
   {
      if (*it)
      {
         (*it)->clear();
         delete *it;
      }
   }
   fCellLists.clear();
   fCellLists.push_back(0);

   TEveProjection::EPType_e pt = fManager->GetProjection()->GetType();
   TEveCaloData::vCellId_t* clv; // ids per phi bin in r-phi projection else ids per eta bins in rho-z projection

   Bool_t isRPhi = (pt == TEveProjection::kPT_RPhi);

   const TAxis* axis = isRPhi ? fData->GetPhiBins() :  fData->GetEtaBins();
   Int_t nBins = axis->GetNbins();

   Float_t min, max;
   if (isRPhi)
   {
      min = GetPhiMin() - fData->GetEps();
      max = GetPhiMax() + fData->GetEps();
      for (Int_t ibin = 1; ibin <= nBins; ++ibin) {
         clv = 0;
         if ( TEveUtil::IsU1IntervalOverlappingByMinMax
              (min, max, axis->GetBinLowEdge(ibin), axis->GetBinUpEdge(ibin)))
         {
            clv = new TEveCaloData::vCellId_t();
            fData->GetCellList(GetEta(), GetEtaRng(), axis->GetBinCenter(ibin), axis->GetBinWidth(ibin), *clv);
            if (!clv->size()) {
               delete clv; clv = 0;
            }
         }
         fCellLists.push_back(clv);
      }
   }
   else
   {
      min = GetEtaMin() - fData->GetEps();
      max = GetEtaMax() + fData->GetEps();
      for (Int_t ibin = 1; ibin <= nBins; ++ibin) {
         clv = 0;
         Float_t low = axis->GetBinLowEdge(ibin);
         Float_t up = axis->GetBinUpEdge(ibin) ;
         if (low >= min && up <= max)
         {
            clv = new TEveCaloData::vCellId_t();
            fData->GetCellList(axis->GetBinCenter(ibin), axis->GetBinWidth(ibin), fPhi, GetPhiRng(), *clv);
            if (!clv->size()) {
               delete clv; clv = 0;
            }
         }
         fCellLists.push_back(clv);
      }
   }

   // cache max bin sum for auto scale
   if (!fScaleAbs)
   {
      fMaxESumBin  = 0;
      fMaxEtSumBin = 0;
      Float_t sumE  = 0;
      Float_t sumEt = 0;
      TEveCaloData::CellData_t  cellData;
      for (Int_t ibin = 1; ibin <= nBins; ++ibin) {
         TEveCaloData::vCellId_t* cids = fCellLists[ibin];
         if (cids)
         {
            sumE = 0; sumEt = 0;
            for (TEveCaloData::vCellId_i it = cids->begin(); it != cids->end(); it++)
            {  
               fData->GetCellData(*it, cellData);
               sumE  += cellData.Value(kFALSE);
               sumEt += cellData.Value(kTRUE);
            }
            fMaxESumBin  = TMath::Max(fMaxESumBin,  sumE);
            fMaxEtSumBin = TMath::Max(fMaxEtSumBin, sumEt);  
         }
      }
      ComputeBBox();
   }

   fCellIdCacheOK= kTRUE;
}

//______________________________________________________________________________
void TEveCalo2D::CellSelectionChanged()
{
   // Sort slected cells in eta or phi bins for selection and highlight.

   CellSelectionChangedInternal(fData->GetCellsSelected(), fCellListsSelected);
   CellSelectionChangedInternal(fData->GetCellsHighlighted(), fCellListsHighlighted);
}

//______________________________________________________________________________
void TEveCalo2D::CellSelectionChangedInternal(TEveCaloData::vCellId_t& inputCells, std::vector<TEveCaloData::vCellId_t*>& outputCellLists)
{
   // Sort slected cells in eta or phi bins.

   Bool_t isRPhi = (fManager->GetProjection()->GetType() == TEveProjection::kPT_RPhi);
   const TAxis* axis = isRPhi ? fData->GetPhiBins() :  fData->GetEtaBins();

   // clear old cache
   for (vBinCells_i it = outputCellLists.begin(); it != outputCellLists.end(); it++)
   {
      if (*it)
      {
         (*it)->clear();
         delete *it;
      }
   }
   outputCellLists.clear();
   UInt_t nBins = axis->GetNbins();
   outputCellLists.resize(nBins+1);
   for (UInt_t b = 0; b <= nBins; ++b)
      outputCellLists[b] = 0;

   for(UInt_t bin = 1; bin <= nBins; ++bin)
   {
      TEveCaloData::vCellId_t* idsInBin = fCellLists[bin];
      if (!idsInBin)
         continue;

      for (TEveCaloData::vCellId_i i = idsInBin->begin(); i != idsInBin->end(); i++)
      { 
         for (TEveCaloData::vCellId_i j = inputCells.begin(); j != inputCells.end(); j++)
         {
            if( (*i).fTower == (*j).fTower && (*i).fSlice == (*j).fSlice)
            {
               if (!outputCellLists[bin])
                  outputCellLists[bin] = new TEveCaloData::vCellId_t();

               outputCellLists[bin]->push_back(TEveCaloData::CellId_t((*i).fTower, (*i).fSlice, (*i).fFraction));
            }
         }
      }
   }
}

//______________________________________________________________________________
void TEveCalo2D::SetScaleAbs(Bool_t sa)
{
   // Set absolute scale in projected calorimeter.
   
   TEveCaloViz::SetScaleAbs(sa);
   BuildCellIdCache();
}

//______________________________________________________________________________
Float_t TEveCalo2D::GetValToHeight() const
{
   // Virtual function of TEveCaloViz.
   // Get transformation factor from E/Et to height.

   AssertCellIdCache();

   if (fScaleAbs)
   {
      return fMaxTowerH/fMaxValAbs;
   }
   else
   {
      if (fData->Empty())
         return 1;

      if (fPlotEt)
         return fMaxTowerH/fMaxEtSumBin;
      else
         return fMaxTowerH/fMaxESumBin;
   }
}

//______________________________________________________________________________
void TEveCalo2D::ComputeBBox()
{
   // Fill bounding-box information of the base-class TAttBBox (virtual method).
   // If member 'TEveFrameBox* fFrame' is set, frame's corners are used as bbox.

   BBoxZero();

   Float_t x, y, z;
   Float_t th = fMaxTowerH                                           ;
   Float_t r  = fBarrelRadius + th;

   x = r,  y = 0, z = 0;
   fManager->GetProjection()->ProjectPoint(x, y, z, fDepth);
   BBoxCheckPoint(x, y, z);
   x = -r, y = 0, z = 0;
   fManager->GetProjection()->ProjectPoint(x, y, z, fDepth);
   BBoxCheckPoint(x, y, z);

   x = 0, y = 0, z = fEndCapPosF + th;
   fManager->GetProjection()->ProjectPoint(x, y, z, fDepth);
   BBoxCheckPoint(x, y, z);
   x = 0, y = 0, z = fEndCapPosB - th;
   fManager->GetProjection()->ProjectPoint(x, y, z, fDepth);
   BBoxCheckPoint(x, y, z);

   x = 0, y = r,  z = 0;
   fManager->GetProjection()->ProjectPoint(x, y, z, fDepth);
   BBoxCheckPoint(x, y, z);
   x = 0, y = -r, z = 0;
   fManager->GetProjection()->ProjectPoint(x, y, z, fDepth);
   BBoxCheckPoint(x, y, z);
}


//==============================================================================
// TEveCaloLego
//==============================================================================

//______________________________________________________________________________
//
// Visualization of calorimeter data as eta/phi histogram.

ClassImp(TEveCaloLego);

//______________________________________________________________________________
TEveCaloLego::TEveCaloLego(TEveCaloData* d, const char* n, const char* t):
   TEveCaloViz(d, n, t),

   fFontColor(-1),
   fGridColor(-1),
   fPlaneColor(kRed-5),
   fPlaneTransparency(60),

   fNZSteps(6),
   fZAxisStep(0.f),

   fAutoRebin(kTRUE),

   fPixelsPerBin(12),
   fNormalizeRebin(kFALSE),

   fProjection(kAuto),
   f2DMode(kValSize),
   fBoxMode(kBack),

   fDrawHPlane(kFALSE),
   fHPlaneVal(0),

   fHasFixedHeightIn2DMode(kFALSE),
   fFixedHeightValIn2DMode(0.f),

   fDrawNumberCellPixels(18), // draw numbers on cell above 30 pixels
   fCellPixelFontSize(12) // size of cell fonts in pixels
{
   // Constructor.

   fMaxTowerH = 1;
   SetElementNameTitle("TEveCaloLego", "TEveCaloLego");
}

//______________________________________________________________________________
void TEveCaloLego::SetData(TEveCaloData* data)
{
   TEveCaloViz::SetData(data);
}

//______________________________________________________________________________
void TEveCaloLego::BuildCellIdCache()
{
   // Build list of drawn cell IDs. For more information see TEveCaloLegoGL:DirectDraw().

   fCellList.clear();

   fData->GetCellList(GetEta(), GetEtaRng(), GetPhi(), GetPhiRng(), fCellList);
   fCellIdCacheOK = kTRUE;
}

//______________________________________________________________________________
void TEveCaloLego::ComputeBBox()
{
   // Fill bounding-box information of the base-class TAttBBox (virtual method).
   // If member 'TEveFrameBox* fFrame' is set, frame's corners are used as bbox.


   // fBBox = Float_t[6] X(min,max), Y(min,max), Z(min,max)

   BBoxZero();

   Float_t ex = 1.2; // 20% offset for axis labels

   Float_t a = 0.5*ex;

   fBBox[0] = -a;
   fBBox[1] =  a;
   fBBox[2] = -a;
   fBBox[3] =  a;

   // scaling is relative to shortest XY axis
   Double_t em, eM, pm, pM;
   fData->GetEtaLimits(em, eM);
   fData->GetPhiLimits(pm, pM);
   Double_t r = (eM-em)/(pM-pm);
   if (r<1)
   {
      fBBox[2] /= r;
      fBBox[3] /= r;
   }
   else
   {
      fBBox[0] *= r;
      fBBox[1] *= r;
   }

   fBBox[4] =  0;
   if (fScaleAbs && !fData->Empty())
      fBBox[5] = GetMaxVal()*GetValToHeight();
   else
      fBBox[5] = fMaxTowerH;
}
