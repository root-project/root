// @(#)root/gui:$Name:  $:$Id: TGWindow.h,v 1.11 2004/02/23 23:49:26 brun Exp $
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

class TGClient;


class TGWindow : public TGObject {

friend class TGClient;

protected:
   const TGWindow   *fParent;         // Parent window
   Bool_t            fNeedRedraw;     // kTRUE if window needs to be redrawn
   TString           fName;           // name of the window used in SavePrimitive()
   static Int_t      fgCounter;       // counter of created windows in SavePrimitive

   TGWindow(Window_t id) : fNeedRedraw(kFALSE) { fClient = 0; fId = id; }

   virtual void DoRedraw() { }

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
   virtual const TGWindow *GetMainFrame() const;

   virtual void MapWindow() { gVirtualX->MapWindow(fId); }
   virtual void MapSubwindows() { gVirtualX->MapSubwindows(fId); }
   virtual void MapRaised() { gVirtualX->MapRaised(fId); }
   virtual void UnmapWindow() { gVirtualX->UnmapWindow(fId); }
   virtual void DestroyWindow() { gVirtualX->DestroyWindow(fId); }
   virtual void RaiseWindow() { gVirtualX->RaiseWindow(fId); }
   virtual void LowerWindow() { gVirtualX->LowerWindow(fId); }
   virtual void IconifyWindow() { gVirtualX->IconifyWindow(fId); }
   virtual void SetBackgroundColor(Pixel_t color)
        { gVirtualX->SetWindowBackground(fId, color); }
   virtual void SetBackgroundPixmap(Pixmap_t pixmap)
        { gVirtualX->SetWindowBackgroundPixmap(fId, pixmap); }

   virtual Bool_t HandleExpose(Event_t *event)
        { if (event->fCount == 0) fClient->NeedRedraw(this); return kTRUE; }
   virtual Bool_t HandleEvent(Event_t *) { return kFALSE; }
   virtual Bool_t HandleTimer(TTimer *) { return kFALSE; }
   virtual void   Move(Int_t x, Int_t y);
   virtual void   Resize(UInt_t w, UInt_t h);
   virtual void   MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual Bool_t IsMapped();
   virtual void   Print(Option_t *option="") const;

   virtual const char *GetName() const { return fName; }
   virtual void        SetName(const char *name) { fName = name; }

   static Int_t        GetCounter();

   ClassDef(TGWindow,0)  // GUI Window base class
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGUnknownWindowHandler                                               //
//                                                                      //
// Handle events for windows that are not part of the native ROOT GUI.  //
// Typically windows created by Xt or Motif (see TRootOIViewer).        //
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
