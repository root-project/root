// @(#)root/gui:$Name:  $:$Id: TGWindow.h,v 1.3 2001/04/11 17:28:08 brun Exp $
// Author: Fons Rademakers   28/12/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGWindow
#define ROOT_TGWindow


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWindow                                                             //
//                                                                      //
// ROOT GUI Window base class.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGObject
#include "TGObject.h"
#endif
#ifndef ROOT_TGClient
#include "TGClient.h"
#endif
#ifndef ROOT_TVirtualX
#include "TVirtualX.h"
#endif

class TTimer;

class TGWindow : public TGObject {

friend class TGClient;
friend class TTimer;

protected:
   const TGWindow   *fParent;         // Parent window
   Bool_t            fNeedRedraw;     // kTRUE if window needs to be redrawn

   TGWindow(Window_t id) : fNeedRedraw(kFALSE) { fClient = 0; fId = id; }

   virtual void DoRedraw() { }
   virtual const TGWindow *GetMainFrame() const
      { return (fParent == fClient->GetRoot()) ? this : fParent->GetMainFrame(); }

public:
   TGWindow(const TGWindow *p, Int_t x, Int_t y,
            UInt_t w, UInt_t h, UInt_t border = 0,
            Int_t depth = 0,
            UInt_t clss = 0,
            void *visual = 0,
            SetWindowAttributes_t *attr = 0,
            UInt_t wtype = 0);
   TGWindow(TGClient *c, Window_t id, const TGWindow *parent = 0);

   virtual ~TGWindow();

   const TGWindow *GetParent() const { return fParent; }

   void MapWindow() { gVirtualX->MapWindow(fId); }
   void MapSubwindows() { gVirtualX->MapSubwindows(fId); }
   void MapRaised() { gVirtualX->MapRaised(fId); }
   void UnmapWindow() { gVirtualX->UnmapWindow(fId); }
   void DestroyWindow() { gVirtualX->DestroyWindow(fId); }
   void RaiseWindow() { gVirtualX->RaiseWindow(fId); }
   void LowerWindow() { gVirtualX->LowerWindow(fId); }
   void IconifyWindow() { gVirtualX->IconifyWindow(fId); }
   void SetBackgroundColor(ULong_t color)
        { gVirtualX->SetWindowBackground(fId, color); }
   void SetBackgroundPixmap(Pixmap_t pixmap)
        { gVirtualX->SetWindowBackgroundPixmap(fId, pixmap); }

   virtual Bool_t HandleExpose(Event_t *event)
        { if (event->fCount == 0) fClient->NeedRedraw(this); return kTRUE; }
   virtual Bool_t HandleEvent(Event_t *) { return kFALSE; }
   virtual Bool_t HandleTimer(TTimer *) { return kFALSE; }
   virtual void   Move(Int_t x, Int_t y);
   virtual void   Resize(UInt_t w, UInt_t h);
   virtual void   MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual Bool_t IsMapped();

   ClassDef(TGWindow,0)  // GUI Window base class
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGUnknownWindowHandler                                               //
//                                                                      //
// Handle events for windows that are not part of the native ROOT GUI.  //
// Typically windows created by Xt or Moptif (see TRootOIViewer).       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGUnknownWindowHandler : public TObject {

public:
   TGUnknownWindowHandler() { }
   virtual ~TGUnknownWindowHandler() { }

   virtual Bool_t HandleEvent(Event_t *) = 0;

   ClassDef(TGUnknownWindowHandler,0)  // Abstract event handler for unknown windows
};

#endif
