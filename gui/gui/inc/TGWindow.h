// @(#)root/gui:$Id$
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


#include "TGObject.h"
#include "TGClient.h"

class TGClient;
class TGIdleHandler;


class TGWindow : public TGObject {

friend class TGClient;

protected:
   const TGWindow   *fParent;         ///< Parent window
   Bool_t            fNeedRedraw;     ///< kTRUE if window needs to be redrawn
   TString           fName;           ///< name of the window used in SavePrimitive()
   static Int_t      fgCounter;       ///< counter of created windows in SavePrimitive
   UInt_t            fEditDisabled;   ///< flags used for "guibuilding"

   TGWindow(Window_t id) :
      fParent(nullptr), fNeedRedraw(kFALSE), fName(), fEditDisabled(0) { fClient = nullptr; fId = id; }
   TGWindow(const TGWindow& tgw) :
      TGObject(tgw), fParent(tgw.fParent), fNeedRedraw(tgw.fNeedRedraw),
      fName(tgw.fName), fEditDisabled(tgw.fEditDisabled) {}

   TGWindow& operator=(const TGWindow& tgw)
   {
      if (this != &tgw) {
         TGObject::operator=(tgw);
         fParent = tgw.fParent;
         fNeedRedraw = tgw.fNeedRedraw;
         fName = tgw.fName;
         fEditDisabled = tgw.fEditDisabled;
      }
      return *this;
   }

   virtual void DoRedraw() { }

public:
   enum  EEditMode {
      kEditEnable        = 0,          ///< allow edit of this window
      kEditDisable       = BIT(0),     ///< disable edit of this window
      kEditDisableEvents = BIT(1),     ///< window events cannot be edited
      kEditDisableGrab   = BIT(2),     ///< window grab cannot be edited
      kEditDisableLayout = BIT(3),     ///< window layout cannot be edited
      kEditDisableResize = BIT(4),     ///< window size cannot be edited
      kEditDisableHeight = BIT(5),     ///< window height cannot be edited
      kEditDisableWidth  = BIT(6),     ///< window width cannot be edited
      kEditDisableBtnEnable = BIT(7),  ///< window can handle mouse button events
      kEditDisableKeyEnable = BIT(8)   ///< window can handle keyboard events
   };

   enum EStatusBits {
      kIsHtmlView = BIT(14)
   };

   TGWindow(const TGWindow *p = nullptr, Int_t x = 0, Int_t y = 0,
            UInt_t w = 0, UInt_t h = 0, UInt_t border = 0,
            Int_t depth = 0,
            UInt_t clss = 0,
            void *visual = nullptr,
            SetWindowAttributes_t *attr = nullptr,
            UInt_t wtype = 0);
   TGWindow(TGClient *c, Window_t id, const TGWindow *parent = nullptr);

   virtual ~TGWindow();

   const TGWindow *GetParent() const { return fParent; }
   virtual const TGWindow *GetMainFrame() const;

   virtual void MapWindow();
   virtual void MapSubwindows();
   virtual void MapRaised();
   virtual void UnmapWindow();
   virtual void DestroyWindow();
   virtual void DestroySubwindows();
   virtual void RaiseWindow();
   virtual void LowerWindow();
   virtual void IconifyWindow();
   virtual void ReparentWindow(const TGWindow *p, Int_t x = 0, Int_t y = 0);
   virtual void RequestFocus();

   virtual void SetBackgroundColor(Pixel_t color);
   virtual void SetBackgroundPixmap(Pixmap_t pixmap);

   virtual Bool_t HandleExpose(Event_t *event)
                  { if (event->fCount == 0) fClient->NeedRedraw(this); return kTRUE; }
   virtual Bool_t HandleEvent(Event_t *) { return kFALSE; }
   Bool_t         HandleTimer(TTimer *) override { return kFALSE; }
   virtual Bool_t HandleIdleEvent(TGIdleHandler *) { return kFALSE; }

   virtual void   Move(Int_t x, Int_t y);
   virtual void   Resize(UInt_t w, UInt_t h);
   virtual void   MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual Bool_t IsMapped();
   virtual Bool_t IsEditable() const { return (fClient->GetRoot() == this); }
   virtual UInt_t GetEditDisabled() const { return fEditDisabled; }
   virtual void   SetEditDisabled(UInt_t on = kEditDisable) { fEditDisabled = on; }
   virtual void   SetEditable(Bool_t on = kTRUE)
                  { if (!(fEditDisabled & kEditDisable)) fClient->SetRoot(on ? this : nullptr); }
   virtual Int_t  MustCleanup() const { return 0; }
   void           Print(Option_t *option="") const override;

   virtual void   SetWindowName(const char *name = nullptr);
   const char    *GetName() const override;
   virtual void   SetName(const char *name) { fName = name; }

   virtual void   SetMapSubwindows(Bool_t /*on*/) {  }
   virtual Bool_t IsMapSubwindows() const { return kTRUE; }

   static Int_t   GetCounter();

   ClassDefOverride(TGWindow, 0);  // GUI Window base class
};


/** \class TGUnknownWindowHandler
    \ingroup guiwidgets

Handle events for windows that are not part of the native ROOT GUI.
Typically windows created by Xt or Motif.

*/


class TGUnknownWindowHandler : public TObject {

public:
   TGUnknownWindowHandler() {}
   virtual ~TGUnknownWindowHandler() {}

   virtual Bool_t HandleEvent(Event_t *) = 0;

   ClassDefOverride(TGUnknownWindowHandler,0)  // Abstract event handler for unknown windows
};

#endif
