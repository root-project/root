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

/** \class TEveCaloViz
\ingroup TEve
Base class for calorimeter data visualization.
See TEveCalo2D and TEveCalo3D for concrete implementations.
*/

ClassImp(TEveCaloViz);

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveCaloViz::~TEveCaloViz()
{
   if (fPalette) fPalette->DecRefCount();
}

////////////////////////////////////////////////////////////////////////////////
/// Get threshold for given slice.

Float_t TEveCaloViz::GetDataSliceThreshold(Int_t slice) const
{
   return fData->RefSliceInfo(slice).fThreshold;
}

////////////////////////////////////////////////////////////////////////////////
/// Management of selection state and ownership of selected cell list
/// is done in TEveCaloData. This is a reason selection is forwarded to it.

TEveElement* TEveCaloViz::ForwardSelection()
{
   return fData;
}

////////////////////////////////////////////////////////////////////////////////
/// Management of selection state and ownership of selected cell list
/// is done in TEveCaloData. We still want GUI editor to display
/// concrete calo-viz object.

TEveElement* TEveCaloViz::ForwardEdit()
{
   return this;
}

////////////////////////////////////////////////////////////////////////////////
/// Set threshold for given slice.

void TEveCaloViz::SetDataSliceThreshold(Int_t slice, Float_t val)
{
   fData->SetSliceThreshold(slice, val);
}

////////////////////////////////////////////////////////////////////////////////
/// Get slice color from data.

Color_t TEveCaloViz::GetDataSliceColor(Int_t slice) const
{
   return fData->RefSliceInfo(slice).fColor;
}

////////////////////////////////////////////////////////////////////////////////
/// Set slice color in data.

void TEveCaloViz::SetDataSliceColor(Int_t slice, Color_t col)
{
   fData->SetSliceColor(slice, col);
}

////////////////////////////////////////////////////////////////////////////////
/// Set eta range.

void TEveCaloViz::SetEta(Float_t l, Float_t u)
{
   fEtaMin=l;
   fEtaMax=u;

   InvalidateCellIdCache();
}

////////////////////////////////////////////////////////////////////////////////
/// Set E/Et plot.

void TEveCaloViz::SetPlotEt(Bool_t isEt)
{
   fPlotEt=isEt;
   if (fPalette)
      fPalette->SetLimits(0, TMath::CeilNint(GetMaxVal()));

   InvalidateCellIdCache();
}

////////////////////////////////////////////////////////////////////////////////

Float_t TEveCaloViz::GetMaxVal() const
{
   // Get maximum plotted value.

   return fData->GetMaxVal(fPlotEt);

}

////////////////////////////////////////////////////////////////////////////////
/// Set phi range.

void TEveCaloViz::SetPhiWithRng(Float_t phi, Float_t rng)
{
   using namespace TMath;

   fPhi = phi;
   fPhiOffset = rng;

   InvalidateCellIdCache();
}

////////////////////////////////////////////////////////////////////////////////
/// Get transition angle between barrel and end-cap cells, assuming fEndCapPosF = -fEndCapPosB.

Float_t TEveCaloViz::GetTransitionTheta() const
{
   return TMath::ATan(fBarrelRadius/fEndCapPosF);
}

////////////////////////////////////////////////////////////////////////////////
/// Get transition eta between barrel and end-cap cells, assuming fEndCapPosF = -fEndCapPosB.

Float_t TEveCaloViz::GetTransitionEta() const
{
   using namespace TMath;
   Float_t t = GetTransitionTheta()*0.5f;
   return -Log(Tan(t));
}

////////////////////////////////////////////////////////////////////////////////
/// Get transition angle between barrel and forward end-cap cells.

Float_t TEveCaloViz::GetTransitionThetaForward() const
{
   return TMath::ATan(fBarrelRadius/fEndCapPosF);
}

////////////////////////////////////////////////////////////////////////////////
/// Get transition eta between barrel and forward end-cap cells.

Float_t TEveCaloViz::GetTransitionEtaForward() const
{
   using namespace TMath;
   Float_t t = GetTransitionThetaForward()*0.5f;
   return -Log(Tan(t));
}

////////////////////////////////////////////////////////////////////////////////
/// Get transition angle between barrel and backward end-cap cells.

Float_t TEveCaloViz::GetTransitionThetaBackward() const
{
   return TMath::ATan(fBarrelRadius/fEndCapPosB);
}

////////////////////////////////////////////////////////////////////////////////
/// Get transition eta between barrel and backward end-cap cells.

Float_t TEveCaloViz::GetTransitionEtaBackward() const
{
   using namespace TMath;
   Float_t t = GetTransitionThetaBackward()*0.5f;
   //negative theta means negative eta
   return Log(-Tan(t));
}


////////////////////////////////////////////////////////////////////////////////
/// Set calorimeter event data.

void TEveCaloViz::SetData(TEveCaloData* data)
{

   if (data == fData) return;
   if (fData) fData->RemoveElement(this);
   fData = data;
   if (fData)
   {
      fData->AddElement(this);
      DataChanged();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Update setting and cache on data changed.
/// Called from TEvecaloData::BroadcastDataChange()

void TEveCaloViz::DataChanged()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Assert cell id cache is ok.
/// Returns true if the cache has been updated.

Bool_t TEveCaloViz::AssertCellIdCache() const
{
   TEveCaloViz* cv = const_cast<TEveCaloViz*>(this);
   if (!fCellIdCacheOK) {
      cv->BuildCellIdCache();
      return kTRUE;
   } else {
      return kFALSE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if given cell is in the ceta phi range.

Bool_t TEveCaloViz::CellInEtaPhiRng(TEveCaloData::CellData_t& cellData) const
{
   if (cellData.EtaMin() >= fEtaMin && cellData.EtaMax() <= fEtaMax)
   {
      if (TEveUtil::IsU1IntervalContainedByMinMax
          (fPhi-fPhiOffset, fPhi+fPhiOffset, cellData.PhiMin(), cellData.PhiMax()))
         return kTRUE;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Assign parameters from given model.

void TEveCaloViz::AssignCaloVizParameters(TEveCaloViz* m)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set TEveRGBAPalette object pointer.

void TEveCaloViz::SetPalette(TEveRGBAPalette* p)
{
   if ( fPalette == p) return;
   if (fPalette) fPalette->DecRefCount();
   fPalette = p;
   if (fPalette) fPalette->IncRefCount();
}

////////////////////////////////////////////////////////////////////////////////
/// Get transformation factor from E/Et to height

Float_t TEveCaloViz::GetValToHeight() const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Make sure the TEveRGBAPalette pointer is not null.
/// If it is not set, a new one is instantiated and the range is set
/// to current min/max signal values.

TEveRGBAPalette* TEveCaloViz::AssertPalette()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Paint this object. Only direct rendering is supported.

void TEveCaloViz::Paint(Option_t* /*option*/)
{
   if (fData)
   {
      PaintStandard(this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveProjectable, returns TEveCalo2D class.

TClass* TEveCaloViz::ProjectedClass(const TEveProjection*) const
{
   return TEveCalo2D::Class();
}

////////////////////////////////////////////////////////////////////////////////
/// Set color and height for a given value and slice using slice color or TEveRGBAPalette.

void TEveCaloViz::SetupColorHeight(Float_t value, Int_t slice, Float_t& outH) const
{
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

/** \class TEveCalo3D
\ingroup TEve
Visualization of a calorimeter event data in 3D.
*/

ClassImp(TEveCalo3D);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveCalo3D::TEveCalo3D(TEveCaloData* d, const char* n, const char* t):
   TEveCaloViz(d, n, t),

   fRnrEndCapFrame    (kTRUE),
   fRnrBarrelFrame    (kTRUE),
   fFrameWidth        (0.5),
   fFrameColor        (kGray+1),
   fFrameTransparency (80)
{
   fCanEditMainColor        = kTRUE;
   fCanEditMainTransparency = kTRUE;
   fMainColorPtr = &fFrameColor;
}

////////////////////////////////////////////////////////////////////////////////
/// Build list of drawn cell IDs. See TEveCalo3DGL::DirectDraw().

void TEveCalo3D::BuildCellIdCache()
{
   fCellList.clear();

   fData->GetCellList(GetEta(), GetEtaRng(), GetPhi(), GetPhiRng(), fCellList);
   fCellIdCacheOK = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill bounding-box information of the base-class TAttBBox (virtual method).
/// If member 'TEveFrameBox* fFrame' is set, frame's corners are used as bbox.

void TEveCalo3D::ComputeBBox()
{
   BBoxInit();

   Float_t th = (fData) ? GetValToHeight() * fData->GetMaxVal(fPlotEt) : 0;

   fBBox[0] = -fBarrelRadius - th;
   fBBox[1] =  fBarrelRadius + th;
   fBBox[2] =  fBBox[0];
   fBBox[3] =  fBBox[1];
   fBBox[4] =  fEndCapPosB - th;
   fBBox[5] =  fEndCapPosF + th;
}

/** \class TEveCalo2D
\ingroup TEve
Visualization of a calorimeter event data in 2D.
*/

ClassImp(TEveCalo2D);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveCalo2D::TEveCalo2D(const char* n, const char* t):
   TEveCaloViz(0, n, t),
   TEveProjected(),
   fOldProjectionType(TEveProjection::kPT_Unknown),
   fMaxESumBin( 0),
   fMaxEtSumBin(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveCalo2D::~TEveCalo2D()
{
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

////////////////////////////////////////////////////////////////////////////////
/// This is virtual method from base-class TEveProjected.

void TEveCalo2D::UpdateProjection()
{
   if (fManager->GetProjection()->GetType() != fOldProjectionType)
   {
      fCellIdCacheOK=kFALSE;
      fOldProjectionType = fManager->GetProjection()->GetType();
   }
   ComputeBBox();
}

////////////////////////////////////////////////////////////////////////////////
/// Set projection manager and model object.

void TEveCalo2D::SetProjection(TEveProjectionManager* mng, TEveProjectable* model)
{
   TEveProjected::SetProjection(mng, model);
   TEveCaloViz* viz = dynamic_cast<TEveCaloViz*>(model);
   AssignCaloVizParameters(viz);
}

////////////////////////////////////////////////////////////////////////////////
/// Build lists of drawn cell IDs. See TEveCalo2DGL::DirecDraw().

void TEveCalo2D::BuildCellIdCache()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Sort selected cells in eta or phi bins for selection and highlight.

void TEveCalo2D::CellSelectionChanged()
{
   CellSelectionChangedInternal(fData->GetCellsSelected(), fCellListsSelected);
   CellSelectionChangedInternal(fData->GetCellsHighlighted(), fCellListsHighlighted);
}

////////////////////////////////////////////////////////////////////////////////
/// Sort selected cells in eta or phi bins.

void TEveCalo2D::CellSelectionChangedInternal(TEveCaloData::vCellId_t& inputCells, std::vector<TEveCaloData::vCellId_t*>& outputCellLists)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set absolute scale in projected calorimeter.

void TEveCalo2D::SetScaleAbs(Bool_t sa)
{
   TEveCaloViz::SetScaleAbs(sa);
   BuildCellIdCache();
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function of TEveCaloViz.
/// Get transformation factor from E/Et to height.

Float_t TEveCalo2D::GetValToHeight() const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Fill bounding-box information of the base-class TAttBBox (virtual method).
/// If member 'TEveFrameBox* fFrame' is set, frame's corners are used as bbox.

void TEveCalo2D::ComputeBBox()
{
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


/** \class TEveCaloLego
\ingroup TEve
Visualization of calorimeter data as eta/phi histogram.
*/

ClassImp(TEveCaloLego);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

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
   fMaxTowerH = 1;
   SetElementNameTitle("TEveCaloLego", "TEveCaloLego");
}

////////////////////////////////////////////////////////////////////////////////
// Set data.

void TEveCaloLego::SetData(TEveCaloData* data)
{
   TEveCaloViz::SetData(data);
}

////////////////////////////////////////////////////////////////////////////////
/// Build list of drawn cell IDs. For more information see TEveCaloLegoGL:DirectDraw().

void TEveCaloLego::BuildCellIdCache()
{
   fCellList.clear();

   fData->GetCellList(GetEta(), GetEtaRng(), GetPhi(), GetPhiRng(), fCellList);
   fCellIdCacheOK = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill bounding-box information of the base-class TAttBBox (virtual method).
/// If member 'TEveFrameBox* fFrame' is set, frame's corners are used as bbox.

void TEveCaloLego::ComputeBBox()
{

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
