// @(#)root/gui:$Id$
// Author: Fons Rademakers   20/9/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGGC
#define ROOT_TGGC


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGGC and TGGCPool                                                    //
//                                                                      //
// Encapsulate a graphics context used in the low level graphics.       //
// TGGCPool provides a pool of graphics contexts.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGObject
#include "TGObject.h"
#endif
#ifndef ROOT_TRefCnt
#include "TRefCnt.h"
#endif

class THashTable;


class TGGC : public TObject, public TRefCnt {

friend class TGGCPool;

protected:
   GCValues_t     fValues;     // graphics context values + mask
   GContext_t     fContext;    // graphics context handle

   TGGC(GCValues_t *values, Bool_t calledByGCPool);
   void UpdateValues(GCValues_t *v);

   TString GetMaskString() const;    //used in SavePrimitive()

public:
   TGGC(GCValues_t *values = 0);
   TGGC(const TGGC &g);
   virtual ~TGGC();
   TGGC &operator=(const TGGC &rhs);

   GContext_t GetGC() const { return fContext; }
   GContext_t operator()() const;

   void SetAttributes(GCValues_t *values);
   void SetFunction(EGraphicsFunction v);
   void SetPlaneMask(ULong_t v);
   void SetForeground(Pixel_t v);
   void SetBackground(Pixel_t v);
   void SetLineWidth(Int_t v);
   void SetLineStyle(Int_t v);
   void SetCapStyle(Int_t v);
   void SetJoinStyle(Int_t v);
   void SetFillStyle(Int_t v);
   void SetFillRule(Int_t v);
   void SetTile(Pixmap_t v);
   void SetStipple(Pixmap_t v);
   void SetTileStipXOrigin(Int_t v);
   void SetTileStipYOrigin(Int_t v);
   void SetFont(FontH_t v);
   void SetSubwindowMode(Int_t v);
   void SetGraphicsExposures(Bool_t v);
   void SetClipXOrigin(Int_t v);
   void SetClipYOrigin(Int_t v);
   void SetClipMask(Pixmap_t v);
   void SetDashOffset(Int_t v);
   void SetDashList(const char v[], Int_t len);
   void SetArcMode(Int_t v);

   const GCValues_t *GetAttributes() const { return &fValues; }
   Mask_t            GetMask() const { return fValues.fMask; }
   EGraphicsFunction GetFunction() const { return fValues.fFunction; }
   ULong_t           GetPlaneMask() const { return fValues.fPlaneMask; }
   Pixel_t           GetForeground() const { return fValues.fForeground; }
   Pixel_t           GetBackground() const { return fValues.fBackground; }
   Int_t             GetLineWidth() const { return fValues.fLineWidth; }
   Int_t             GetLineStyle() const { return fValues.fLineStyle; }
   Pixmap_t          GetTile() const { return fValues.fTile; }
   Pixmap_t          GetStipple() const { return fValues.fStipple; }
   Int_t             GetTileStipXOrigin() const { return fValues.fTsXOrigin; }
   Int_t             GetTileStipYOrigin() const { return fValues.fTsYOrigin; }
   Int_t             GetSubwindowMode() const { return fValues.fSubwindowMode; }
   FontH_t           GetFont() const { return fValues.fFont; }
   Bool_t            GetGraphicsExposures() const { return fValues.fGraphicsExposures; }
   Int_t             GetClipXOrigin() const { return fValues.fClipXOrigin; }
   Int_t             GetClipYOrigin() const { return fValues.fClipYOrigin; }
   Pixmap_t          GetClipMask() const { return fValues.fClipMask; }
   Int_t             GetCapStyle() const { return fValues.fCapStyle; }
   Int_t             GetJoinStyle() const { return fValues.fJoinStyle; }
   Int_t             GetFillStyle() const { return fValues.fFillStyle; }
   Int_t             GetFillRule() const { return fValues.fFillRule; }
   Int_t             GetDashOffset() const { return fValues.fDashOffset; }
   Int_t             GetDashLen() const { return fValues.fDashLen; }
   const char       *GetDashes() const { return fValues.fDashes; }
   Int_t             GetArcMode() const { return fValues.fArcMode; }

   void Print(Option_t *option="") const;
   void SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGGC,0)  // Graphics context
};


class TGGCPool : public TGObject {

friend class TGGC;

private:
   THashTable  *fList;   // hash table of graphics contexts in pool

   void   ForceFreeGC(const TGGC *gc);
   Int_t  MatchGC(const TGGC *gc, GCValues_t *values);
   void   UpdateGC(TGGC *gc, GCValues_t *values);

protected:
   TGGCPool(const TGGCPool& gp) : TGObject(gp), fList(gp.fList) { }
   TGGCPool& operator=(const TGGCPool& gp)
     {if(this!=&gp) {TGObject::operator=(gp); fList=gp.fList;}
     return *this;}

public:
   TGGCPool(TGClient *client);
   virtual ~TGGCPool();

   TGGC *GetGC(GCValues_t *values, Bool_t rw = kFALSE);
   TGGC *GetGC(GContext_t gct);
   void  FreeGC(const TGGC *gc);
   void  FreeGC(GContext_t gc);

   TGGC *FindGC(const TGGC *gc);
   TGGC *FindGC(GContext_t gc);

   void  Print(Option_t *option="") const;

   ClassDef(TGGCPool,0)  // Graphics context pool
};

#endif
