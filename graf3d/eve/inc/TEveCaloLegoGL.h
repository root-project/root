// @(#)root/eve:$Id$
// Author: Alja Mrak-Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveCaloLegoGL
#define ROOT_TEveCaloLegoGL

#include "TGLObject.h"
#include "TGLAxisPainter.h"

#include "TEveCaloData.h"
#include "TEveVSDStructs.h"
#include "TEveCalo.h"

#include <map>

class TEveCaloLego;

class TEveCaloLegoGL : public TGLObject
{
private:
   TEveCaloLegoGL(const TEveCaloLegoGL&);            // Not implemented
   TEveCaloLegoGL& operator=(const TEveCaloLegoGL&); // Not implemented

   mutable Float_t   fDataMax; // cached

   mutable Color_t   fGridColor; // cached
   mutable Color_t   fFontColor; // cached

   // axis
   mutable TAxis    *fEtaAxis;
   mutable TAxis    *fPhiAxis;
   mutable TAxis    *fZAxis;

   mutable TEveVector  fXAxisTitlePos;
   mutable TEveVector  fYAxisTitlePos;
   mutable TEveVector  fZAxisTitlePos;
   mutable TEveVector  fBackPlaneXConst[2];
   mutable TEveVector  fBackPlaneYConst[2];

   mutable TGLAxisPainter fAxisPainter;

protected:
   Int_t   GetGridStep(TGLRnrCtx &rnrCtx) const;
   void    RebinAxis(TAxis *orig, TAxis *curr) const;

   void    SetAxis3DTitlePos(TGLRnrCtx &rnrCtx, Float_t x0, Float_t x1, Float_t y0, Float_t y1) const;
   void    DrawAxis3D(TGLRnrCtx &rnrCtx) const;
   void    DrawAxis2D(TGLRnrCtx &rnrCtx) const;
   void    DrawHistBase(TGLRnrCtx &rnrCtx) const;

   void    DrawCells2D(TGLRnrCtx & rnrCtx) const;

   void    DrawCells3D(TGLRnrCtx & rnrCtx) const;
   void    MakeQuad(Float_t x, Float_t y, Float_t z, Float_t xw, Float_t yw, Float_t zh) const;
   void    MakeDisplayList() const;

   void    WrapTwoPi(Float_t &min, Float_t &max) const;

   TEveCaloLego                     *fM;  // Model object.
   mutable Bool_t                    fDLCacheOK;

   typedef std::map<Int_t, UInt_t>           SliceDLMap_t;
   typedef std::map<Int_t, UInt_t>::iterator SliceDLMap_i;

   mutable SliceDLMap_t              fDLMap;
   mutable TEveCaloData::RebinData_t fRebinData;

   mutable Bool_t                    fCells3D;
public:
   TEveCaloLegoGL();
   virtual ~TEveCaloLegoGL();

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt = 0);

   virtual void   SetBBox();

   virtual void   DLCacheDrop();
   virtual void   DLCachePurge();

   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;
   virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   virtual void   ProcessSelection(TGLRnrCtx & rnrCtx, TGLSelectRecord & rec);

   ClassDef(TEveCaloLegoGL, 0); // GL renderer class for TEveCaloLego.
};

//______________________________________________________________________________
inline void TEveCaloLegoGL::WrapTwoPi(Float_t &min, Float_t &max) const
{
   if (fM->GetData()->GetWrapTwoPi())
   {
      if (fM->GetPhiMax()>TMath::Pi() && max<=fM->GetPhiMin())
      {
         min += TMath::TwoPi();
         max += TMath::TwoPi();
      }
      else if (fM->GetPhiMin()<-TMath::Pi() && min>=fM->GetPhiMax())
      {
         min -= TMath::TwoPi();
         max -= TMath::TwoPi();
      }
   }
}
#endif
