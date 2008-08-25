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
   Float_t  fTransTheta; // transition theta
   Float_t  fTheta;

   TEveCaloLego*  fLastPickedLego;

   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2);

public:
   TEveLegoEventHandler(const char *name, TGWindow *w, TObject *obj, const char *title="");
   virtual ~TEveLegoEventHandler() {}

   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleDoubleClick(Event_t *event);

   Float_t GetTransTheta() {return fTransTheta;}
   void    SetTransTheta(Float_t h) {fTransTheta=h;}

   ClassDef(TEveLegoEventHandler, 0); // A GL event handler class. Swiches perspective or orthographic camera.
};

#endif
