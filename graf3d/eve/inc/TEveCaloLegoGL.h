// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

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
#include "TEveCaloData.h"
#include "TGLIncludes.h"
#include "TGLAxisPainter.h"
#include <map>

class TEveCaloLego;

class TEveCaloLegoGL : public TGLObject
{
private:
   TEveCaloLegoGL(const TEveCaloLegoGL&);            // Not implemented
   TEveCaloLegoGL& operator=(const TEveCaloLegoGL&); // Not implemented

   mutable Float_t   fDataMax; // cached

   mutable TAxis*    fEtaAxis;
   mutable TAxis*    fPhiAxis;
   mutable Int_t     fBinStep;

   mutable TGLAxisAttrib    fXAxisAtt;
   mutable TGLAxisAttrib    fYAxisAtt;
   mutable TGLAxisAttrib    fZAxisAtt;

   mutable TGLAxisPainter   fAxisPainter;
protected:
   Int_t   GetGridStep(TGLRnrCtx &rnrCtx) const;
   void    SetAxis(TAxis *orig, TAxis *curr) const;

   Bool_t  PhiShiftInterval(Float_t &min, Float_t &max) const;

   void    DrawZScales3D(TGLRnrCtx &rnrCtx, Float_t x0, Float_t x1, Float_t y0, Float_t y1) const;
   void    DrawZAxis(TGLRnrCtx &rnrCtx, Float_t azX, Float_t azY) const;

   void    DrawXYScales(TGLRnrCtx &rnrCtx, Float_t x0, Float_t x1, Float_t y0, Float_t y1) const;
   void    DrawHistBase(TGLRnrCtx &rnrCtx) const;

   void    DrawCells2D() const;

   void    DrawCells3D(TGLRnrCtx & rnrCtx) const;
   void    MakeQuad(Float_t x, Float_t y, Float_t z,
                    Float_t xw, Float_t yw, Float_t zh) const;
   void    MakeDisplayList() const;

   mutable Bool_t                    fDLCacheOK;

   typedef std::map<Int_t, UInt_t>           SliceDLMap_t;
   typedef std::map<Int_t, UInt_t>::iterator SliceDLMap_i;

   mutable SliceDLMap_t              fDLMap;
   mutable TEveCaloData::RebinData_t fRebinData;

   mutable Bool_t           fCells3D;

   TEveCaloLego            *fM;  // Model object.

public:
   TEveCaloLegoGL();
   virtual ~TEveCaloLegoGL();

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);

   virtual void   SetBBox();

   virtual void   DLCacheDrop();
   virtual void   DLCachePurge();

   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;

   virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   virtual void   ProcessSelection(TGLRnrCtx & rnrCtx, TGLSelectRecord & rec);

   ClassDef(TEveCaloLegoGL, 0); // GL renderer class for TEveCaloLego.
};

#endif
