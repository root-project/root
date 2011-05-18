// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TEveLegoEventHandler
#define ROOT_TEveLegoEventHandler

#include "TGLEventHandler.h"
#include "TGLCamera.h"

class TEveCaloLego;

class TEveLegoEventHandler : public TGLEventHandler
{
private:
   TEveLegoEventHandler(const TEveLegoEventHandler&);            // Not implemented
   TEveLegoEventHandler& operator=(const TEveLegoEventHandler&); // Not implemented

protected:
   enum EMode_e   { kLocked, kFree };

   EMode_e  fMode;       // current rotation mode
   Float_t  fTransTheta; // transition theta in radians
   Float_t  fTheta;

   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2);

public:
   TEveCaloLego*  fLego;

   TEveLegoEventHandler(TGWindow *w, TObject *obj, TEveCaloLego* lego = 0);
   virtual ~TEveLegoEventHandler() {}

   virtual Bool_t HandleKey(Event_t *event);

   Float_t GetTransTheta() {return fTransTheta;}
   void    SetTransTheta(Float_t h) {fTransTheta=h;}

   TEveCaloLego* GetLego() { return fLego; }
   void          SetLego( TEveCaloLego* x) { fLego = x; }

   ClassDef(TEveLegoEventHandler, 0); // A GL event handler class. Swiches perspective or orthographic camera.
};

#endif
