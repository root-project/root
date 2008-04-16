// @(#)root/gui:$Id$
// Author: Abdelhalim Ssadik   07/07/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGDockableFrame
#define ROOT_TGDockableFrame


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TGDockableFrame is a frame with handles that allow it to be        //
// undocked (i.e. put in a transient frame of its own) and to be docked //
// again or hidden and shown again. It uses the TGDockButton, which is  //
// a button with two vertical bars (||) and TGDockHideButton, which is  //
// a button with a small triangle. The TGUndockedFrame is a transient   //
// frame that on closure will put the frame back in the dock.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif

#ifndef ROOT_TGButton
#include "TGButton.h"
#endif

#ifndef ROOT_TGWindow
#include "TGWindow.h"
#endif


class TGDockableFrame;


class TGDockButton : public TGButton {
protected:
   Bool_t     fMouseOn;    // true when mouse on button
   ULong_t    fNormBg;     // normal background color
   ULong_t    fHiBg;       // highlighted background color

   virtual void DrawBorder();
   virtual void DoRedraw();

public:
   TGDockButton(const TGCompositeFrame *p = 0, Int_t id = 1);
   virtual ~TGDockButton();

   virtual Bool_t HandleCrossing(Event_t *event);

   ClassDef(TGDockButton,0)  // Dock button
};


class TGDockHideButton : public TGDockButton {
protected:
   Int_t     fAspectRatio;   // triangle orientation

   virtual void DoRedraw();

public:
   TGDockHideButton(const TGCompositeFrame *p = 0);

   void SetAspectRatio(Int_t a) { fAspectRatio = a; DoRedraw(); }

   ClassDef(TGDockHideButton,0)  // Hide dock button
};


class TGUndockedFrame : public TGTransientFrame {

private:
   TGUndockedFrame(const TGUndockedFrame&); // Not implemented
   TGUndockedFrame& operator=(const TGUndockedFrame&); // Not implemented

protected:
   TGDockableFrame    *fDockable;   // orignal dockable frame

public:
   TGUndockedFrame(const TGWindow *p = 0, TGDockableFrame *dockable = 0);
   virtual ~TGUndockedFrame();

   void FixSize();
   void CloseWindow();

   ClassDef(TGUndockedFrame,0)  // Undocked frame
};


class TGDockableFrame : public TGCompositeFrame, public TGWidget {
friend class TGUndockedFrame;

private:
   TGDockableFrame(const TGDockableFrame&); // Not implemented
   TGDockableFrame& operator=(const TGDockableFrame&); // Not implemented

protected:
   Bool_t            fHidden;        // if frame is hidden
   Bool_t            fEnableHide;    // if frame can be hidden
   Bool_t            fEnableUndock;  // if frame can be undocked
   Bool_t            fDeleted;       // kTRUE if it is being deleted
   Bool_t            fFixedSize;     // kTRUE if fixed size when undocked
   TString           fDockName;      // name of frame
   TGCompositeFrame *fContainer;     // container containing dockable frame
   TGCompositeFrame *fButtons;       // container containing dock and hide buttons
   TGDockButton     *fDockButton;    // dock button
   TGDockHideButton *fHideButton;    // hide button
   TGUndockedFrame  *fFrame;         // undocked frame
   TGLayoutHints    *fCl, *fHints;   // layout hints
   TGLayoutHints    *fLb, *fLc;      // layout hints

public:
   TGDockableFrame(const TGWindow *p = 0, Int_t id = -1,
                   UInt_t options = kHorizontalFrame);
   virtual ~TGDockableFrame();

   virtual void AddFrame(TGFrame *f, TGLayoutHints *hints);

   virtual Bool_t ProcessMessage(Long_t, Long_t, Long_t);
   virtual void Docked() { Emit("Docked()"); }        //*SIGNAL*
   virtual void Undocked() { Emit("Undocked()"); }    //*SIGNAL*

   void UndockContainer();
   void DockContainer(Int_t del = kTRUE);

   void HideContainer();
   void ShowContainer();

   void   EnableUndock(Bool_t onoff);
   Bool_t EnableUndock() const { return fEnableUndock; }
   void   EnableHide(Bool_t onoff);
   Bool_t EnableHide() const { return fEnableHide; }

   void SetWindowName(const char *name);

   Bool_t IsUndocked() const { return (fFrame != 0); }
   Bool_t IsHidden() const { return fHidden; }

   Bool_t IsFixedSize() const { return  fFixedSize; }
   void   SetFixedSize(Bool_t fixed) { fFixedSize = fixed; }

   TGCompositeFrame *GetContainer() const { return fContainer; }
   TGUndockedFrame  *GetUndocked() const { return fFrame; }

   virtual void      SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TGDockableFrame,0)  // Dockable widget
};

#endif
