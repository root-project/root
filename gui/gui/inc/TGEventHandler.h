// @(#)root/gui:$Id: TGEventHandler.h
// Author: Bertrand Bellenot   29/01/2008

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGEventHandler
#define ROOT_TGEventHandler


#include "TNamed.h"
#include "TQObject.h"
#include "GuiTypes.h"

class TGWindow;

class TGEventHandler : public TNamed, public TQObject {

private:

   TGEventHandler(const TGEventHandler&) = delete;
   TGEventHandler& operator=(const TGEventHandler&) = delete;

   Bool_t   fIsActive;    ///< kTRUE if handler is active, kFALSE if not active
   TGWindow *fWindow;
   TObject  *fObject;

   void  *GetSender() override { return this; }  //used to set gTQSender

public:
   TGEventHandler(const char *name, TGWindow *w, TObject *obj, const char *title="") :
      TNamed(name, title), fIsActive(kTRUE), fWindow(w), fObject(obj) { }
   virtual ~TGEventHandler() { }

   void           Activate() { fIsActive = kTRUE; }
   void           DeActivate() { fIsActive = kFALSE; }
   Bool_t         IsActive() const { return fIsActive; }
   virtual Bool_t HandleEvent(Event_t *ev);
   virtual Bool_t HandleConfigureNotify(Event_t *) { return kFALSE; }
   virtual Bool_t HandleButton(Event_t *) { return kFALSE; }
   virtual Bool_t HandleDoubleClick(Event_t *) { return kFALSE; }
   virtual Bool_t HandleCrossing(Event_t *) { return kFALSE; }
   virtual Bool_t HandleMotion(Event_t *) { return kFALSE; }
   virtual Bool_t HandleKey(Event_t *) { return kFALSE; }
   virtual Bool_t HandleFocusChange(Event_t *) { return kFALSE; }
   virtual Bool_t HandleSelection(Event_t *) { return kFALSE; }
   virtual Bool_t HandleSelectionRequest(Event_t *) { return kFALSE; }
   virtual Bool_t HandleSelectionClear(Event_t *) { return kFALSE; }
   virtual Bool_t HandleColormapChange(Event_t *) { return kFALSE; }
   virtual void   ProcessedEvent(Event_t *event)
                     { Emit("ProcessedEvent(Event_t*)", (Longptr_t)event); } //*SIGNAL*

   virtual void   SendMessage(const TGWindow *w, Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
   virtual Bool_t ProcessMessage(Longptr_t, Longptr_t, Longptr_t) { return kFALSE; }
   virtual void   Repaint() { }

   ClassDefOverride(TGEventHandler,0)  // Abstract event handler
};

#endif

