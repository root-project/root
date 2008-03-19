// @(#)root/gui:$Id: TGEventHandler.h
// Author: Bertrand Bellenot   29/01/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGEventHandler
#define ROOT_TGEventHandler

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGEventHandler                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif
#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif

class TGWindow;

class TGEventHandler : public TNamed, public TQObject {

private:
  
   TGEventHandler(const TGEventHandler&); // Not implemented
   TGEventHandler& operator=(const TGEventHandler&); // Not implemented

   Bool_t   fIsActive;    // kTRUE if handler is active, kFALSE if not active
   TGWindow *fWindow;
   TObject  *fObject;

   void  *GetSender() { return this; }  //used to set gTQSender

public:
   TGEventHandler(const char *name, TGWindow *w, TObject *obj, const char *title="") : 
      TNamed(name, title), fIsActive(kTRUE), fWindow(w), fObject(obj) { };
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
                     { Emit("ProcessedEvent(Event_t*)", (Long_t)event); } //*SIGNAL*

   virtual void   SendMessage(const TGWindow *w, Long_t msg, Long_t parm1, Long_t parm2);
   virtual Bool_t ProcessMessage(Long_t, Long_t, Long_t) { return kFALSE; }
   virtual void   Repaint() { }

   ClassDef(TGEventHandler,0)  // Abstract event handler
};

#endif

