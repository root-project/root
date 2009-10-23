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
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"
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
   fEndCapPos(-1.f),

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
TEveElement* TEveCaloViz::ForwardSelection() const
{
   // Management of selection state and ownershih of selected cell list
   // is done in TEveCaloData. This is a reason selection is forwared to it.

   return fData;
}

//______________________________________________________________________________
void TEveCaloViz::IncImpliedSelected()
{
   // Virtual method od TEveElement::IncImpliedSelected().
   // It has same functionality as its base class with additional
   // debug print of selected cells list.
   if (gDebug > 1)
   {
      printf("%s::IncImpliedSelected, selected %d cells:\n", GetElementName(), (Int_t)fData->GetCellsSelected().size());
      TEveCaloData::CellData_t cellData;
      TEveCaloData::vCellId_t& sel = fData->GetCellsSelected();
      for (TEveCaloData::vCellId_i it = sel.begin(); it != sel.end(); ++it)
      {
         fData->GetCellData((*it), cellData);
         printf("Tower [%d] Slice [%d] Value [%.2f] ", (*it).fTower, (*it).fSlice, cellData.fValue);
         printf("Eta:(%f, %f) Phi(%f, %f)\n",  cellData.fEtaMin, cellData.fEtaMax, cellData.fPhiMin, cellData.fPhiMax);
      }
   }

   TEveElement::IncImpliedSelected();
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
      fPalette->SetLimits(0, TMath::CeilNint(fData->GetMaxVal(fPlotEt)));

   InvalidateCellIdCache();
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetMaxVal() const
{

   // Get maximum plotted value.

   if (fScaleAbs)
      return fMaxValAbs;
   else
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
   // Get transition angle between barrel and end-cap cells.

   return TMath::ATan(fBarrelRadius/fEndCapPos);
}

//______________________________________________________________________________
Float_t TEveCaloViz::GetTransitionEta() const
{
   // Get transition eta between barrel and end-cap cells.

   using namespace TMath;
   Float_t t = GetTransitionTheta()*0.5f;
   return -Log(Tan(t));
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
      Int_t hlimit = TMath::CeilNint(fScaleAbs ? fMaxValAbs : fData->GetMaxVal(fPlotEt));
      fPalette->SetLimits(0, hlimit);
      fPalette->SetMin(0);
      fPalette->SetMax(hlimit);
   }

   InvalidateCellIdCache();
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
   fEndCapPos    = m->fEndCapPos;

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

      Int_t hlimit = TMath::CeilNint(fScaleAbs ? fMaxValAbs : fData->GetMaxVal(fPlotEt));
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

   static const TEveException eH("TEvecaloViz::Paint ");

   if (!fData)
      return;

   TBuffer3D buff(TBuffer3DTypes::kGeneric);

   // Section kCore
   buff.fID           = this;
   buff.fColor        = GetMainColor();
   buff.fTransparency = GetMainTransparency();
   if (HasMainTrans())
      RefMainTrans().SetBuffer3D(buff);
   buff.SetSectionsValid(TBuffer3D::kCore);

   Int_t reqSections = gPad->GetViewer3D()->AddObject(buff);
   if (reqSections != TBuffer3D::kNone)
      Error(eH, "only direct GL rendering supported.");
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
      TGLUtil::Color4ubv(c);
   }
   else
   {
      TGLUtil::Color(fData->RefSliceInfo(slice).fColor);
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
   fFrameColor        (kGray+1),
   fFrameTransparency (80)
{

   // Constructor.

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
   fBBox[4] = -fEndCapPos - th;
   fBBox[5] =  fEndCapPos + th;
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
   fOldProjectionType(TEveProjection::kPT_Unknown)
{
   // Constructor.
}

//______________________________________________________________________________
TEveCalo2D::~TEveCalo2D()
{
   // Destructor.

   for(UInt_t vi = 0; vi < fCellLists.size(); ++vi)
   {
      TEveCaloData::vCellId_t* cids = fCellLists[vi];
      cids->clear();
      delete cids;
   }

   for(UInt_t vi = 0; vi < fCellListsSelected.size(); ++vi)
   {
      TEveCaloData::vCellId_t* cids = fCellListsSelected[vi];
      cids->clear();
      delete cids;
   }
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
      delete *it;
   fCellLists.clear();

   TEveProjection::EPType_e pt = fManager->GetProjection()->GetType();
   TEveCaloData::vCellId_t*  clv; // ids per phi bin in r-phi projection else ids per eta bins in rho-z projection

   if (pt == TEveProjection::kPT_RPhi)
   {
      // build list on basis of phi bins
      const TAxis* ay = fData->GetPhiBins();
      assert(ay);
      Int_t nBins = ay->GetNbins();
      for (Int_t ibin = 1; ibin <= nBins; ++ibin)
      {
         if ( TEveUtil::IsU1IntervalOverlappingByMinMax
              (GetPhiMin(), GetPhiMax(), ay->GetBinLowEdge(ibin), ay->GetBinUpEdge(ibin)))
         {
            clv = new TEveCaloData::vCellId_t();
            fData->GetCellList(GetEta(), GetEtaRng(), ay->GetBinCenter(ibin), ay->GetBinWidth(ibin), *clv);
            if (clv->size())
               fCellLists.push_back(clv);
            else
               delete clv;
         }
      }
   }
   else if (pt == TEveProjection::kPT_RhoZ)
   {
      // build list on basis of eta bins
      const TAxis *ax    = fData->GetEtaBins();
      assert(ax);
      const Int_t  nBins = ax->GetNbins();
      for (Int_t ibin = 1; ibin <= nBins; ++ibin)
      {
         if (ax->GetBinLowEdge(ibin) > fEtaMin && ax->GetBinUpEdge(ibin) <= fEtaMax)
         {
            clv = new TEveCaloData::vCellId_t();
            fData->GetCellList(ax->GetBinCenter(ibin), ax->GetBinWidth(ibin), fPhi, GetPhiRng(), *clv);
            if (clv->size())
               fCellLists.push_back(clv);
            else
               delete clv;
         }
      }
   }

   BuildCellIdCacheSelected();

   fCellIdCacheOK= kTRUE;
}

//______________________________________________________________________________
void TEveCalo2D::BuildCellIdCacheSelected()
{
   // Sort slected cells in eta or phi bins.

   // clear old cache
   for (vBinCells_i it = fCellListsSelected.begin(); it != fCellListsSelected.end(); it++)
      delete *it;
   fCellListsSelected.clear();

   TEveCaloData::CellData_t cellData;
   UInt_t ncs = fData->GetCellsSelected().size();
   if (ncs)
   {
      Bool_t rPhi  = fManager->GetProjection()->GetType() == TEveProjection::kPT_RPhi;
      UInt_t nBins = rPhi ? fData->GetPhiBins()->GetNbins() : fData->GetEtaBins()->GetNbins();

      fCellListsSelected.resize(nBins);
      for (UInt_t vi = 0; vi < nBins; ++vi)
         fCellListsSelected[vi] = 0;

      Int_t bin;
      for (UInt_t i=0; i < ncs; i++)
      {
         fData->GetCellData(fData->GetCellsSelected()[i], cellData);
         if (rPhi)
            bin = fData->GetPhiBins()->FindBin(cellData.Phi());
         else
            bin = fData->GetEtaBins()->FindBin(cellData.Eta());

         if (fCellListsSelected[bin] == 0)
            fCellListsSelected[bin] = new TEveCaloData::vCellId_t();

         fCellListsSelected[bin]->push_back(fData->GetCellsSelected()[i]);
      }
   }
}

//______________________________________________________________________________
void TEveCalo2D::ComputeBBox()
{
   // Fill bounding-box information of the base-class TAttBBox (virtual method).
   // If member 'TEveFrameBox* fFrame' is set, frame's corners are used as bbox.

   BBoxZero();

   Float_t x, y, z;
   Float_t th = fData ? GetValToHeight()*fData->GetMaxVal(fPlotEt) :0;
   Float_t r  = fBarrelRadius + th;
   Float_t ze = fEndCapPos + th;

   x = r,  y = 0, z = 0;
   fManager->GetProjection()->ProjectPoint(x, y, z, fDepth);
   BBoxCheckPoint(x, y, z);
   x = -r, y = 0, z = 0;
   fManager->GetProjection()->ProjectPoint(x, y, z, fDepth);
   BBoxCheckPoint(x, y, z);

   x = 0, y = 0, z = ze;
   fManager->GetProjection()->ProjectPoint(x, y, z, fDepth);
   BBoxCheckPoint(x, y, z);
   x = 0, y = 0, z = -ze;
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

   fPixelsPerBin(9),
   fNormalizeRebin(kTRUE),

   fProjection(kAuto),
   f2DMode(kValColor),
   fBoxMode(kBack),

   fDrawHPlane(kFALSE),
   fHPlaneVal(0),

   fBinStep(-1),
   fDrawNumberCellPixels(8), // draw numbers on cell above 30 pixels
   fCellPixelFontSize(15) // size of cell fonts in pixels
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

   Float_t ex = 1.2;

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
   if (fScaleAbs)
      fBBox[5] = fMaxTowerH;
   else
      fBBox[5] = 1;
}
