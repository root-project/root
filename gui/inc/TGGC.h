// @(#)root/gui:$Name:  $:$Id: TGGC.h,v 1.1 2000/09/29 08:52:52 rdm Exp $
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

class TList;


class TGGC : public TObject {

friend class TGGCPool;

protected:
   GCValues_t     fValues;     // graphics context values + mask
   GContext_t     fContext;    // graphics context handle
   Bool_t         fDelete;     // if true delete graphics context

   void UpdateValues(GCValues_t *v);

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
   void SetForeground(ULong_t v);
   void SetBackground(ULong_t v);
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
   void SetDashList(char v[], Int_t len);
   void SetArcMode(Int_t v);

   ClassDef(TGGC,0)  // Graphics context
};


class TGGCPool : public TGObject {

friend class TGClient;

protected:
   class TGGCElement : public TObject, public TRefCnt {
   public:
      TGGC   *fContext;
      ~TGGCElement() { delete fContext; }
      Bool_t  IsEqual(TObject *obj) { return fContext == obj; }
   };
   TList  *fList;   // list of graphics contexts in pool

private:
   Int_t  MatchGC(TGGC *gc, GCValues_t *values);
   void   UpdateGC(TGGC *gc, GCValues_t *values);

public:
   TGGCPool(TGClient *client);
   virtual ~TGGCPool();

   TGGC *GetGC(GCValues_t *values);
   void  FreeGC(TGGC *gc);

   ClassDef(TGGCPool,0)  // Graphics context pool
};

#endif
