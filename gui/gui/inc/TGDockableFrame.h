// @(#)root/gui:$Id$
// Author: Abdelhalim Ssadik   07/07/04

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGDockableFrame
#define ROOT_TGDockableFrame


#include "TGFrame.h"

#include "TGWidget.h"

#include "TGButton.h"

#include "TGWindow.h"


class TGDockableFrame;


class TGDockButton : public TGButton {
protected:
   Bool_t     fMouseOn;    ///< true when mouse on button
   ULong_t    fNormBg;     ///< normal background color
   ULong_t    fHiBg;       ///< highlighted background color

   void DrawBorder() override;
   void DoRedraw() override;

public:
   TGDockButton(const TGCompositeFrame *p = nullptr, Int_t id = 1);
   virtual ~TGDockButton();

   Bool_t HandleCrossing(Event_t *event) override;

   ClassDefOverride(TGDockButton,0)  // Dock button
};


class TGDockHideButton : public TGDockButton {
protected:
   Int_t     fAspectRatio;   ///< triangle orientation

   void DoRedraw() override;

public:
   TGDockHideButton(const TGCompositeFrame *p = nullptr);

   void SetAspectRatio(Int_t a) { fAspectRatio = a; DoRedraw(); }

   ClassDefOverride(TGDockHideButton,0)  // Hide dock button
};


class TGUndockedFrame : public TGTransientFrame {

private:
   TGUndockedFrame(const TGUndockedFrame&) = delete;
   TGUndockedFrame& operator=(const TGUndockedFrame&) = delete;

protected:
   TGDockableFrame    *fDockable;   ///< orignal dockable frame

public:
   TGUndockedFrame(const TGWindow *p = nullptr, TGDockableFrame *dockable = nullptr);
   virtual ~TGUndockedFrame();

   void FixSize();
   void CloseWindow() override;

   ClassDefOverride(TGUndockedFrame,0)  // Undocked frame
};


class TGDockableFrame : public TGCompositeFrame, public TGWidget {
friend class TGUndockedFrame;

private:
   TGDockableFrame(const TGDockableFrame&) = delete;
   TGDockableFrame& operator=(const TGDockableFrame&) = delete;

protected:
   Bool_t            fHidden;        ///< if frame is hidden
   Bool_t            fEnableHide;    ///< if frame can be hidden
   Bool_t            fEnableUndock;  ///< if frame can be undocked
   Bool_t            fDeleted;       ///< kTRUE if it is being deleted
   Bool_t            fFixedSize;     ///< kTRUE if fixed size when undocked
   TString           fDockName;      ///< name of frame
   TGCompositeFrame *fContainer;     ///< container containing dockable frame
   TGCompositeFrame *fButtons;       ///< container containing dock and hide buttons
   TGDockButton     *fDockButton;    ///< dock button
   TGDockHideButton *fHideButton;    ///< hide button
   TGUndockedFrame  *fFrame;         ///< undocked frame
   TGLayoutHints    *fHints;         ///< layout hints
   TGLayoutHints    *fLb, *fLc;      ///< layout hints

public:
   TGDockableFrame(const TGWindow *p = nullptr, Int_t id = -1,
                   UInt_t options = kHorizontalFrame);
   virtual ~TGDockableFrame();

   void AddFrame(TGFrame *f, TGLayoutHints *hints) override;

   Bool_t ProcessMessage(Longptr_t, Longptr_t, Longptr_t) override;
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

   void SetWindowName(const char *name) override;

   Bool_t IsUndocked() const { return fFrame != nullptr; }
   Bool_t IsHidden() const { return fHidden; }

   Bool_t IsFixedSize() const { return  fFixedSize; }
   void   SetFixedSize(Bool_t fixed) { fFixedSize = fixed; }

   TGCompositeFrame *GetContainer() const { return fContainer; }
   TGUndockedFrame  *GetUndocked() const { return fFrame; }

   void      SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGDockableFrame,0)  // Dockable widget
};

#endif
