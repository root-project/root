// @(#)root/gui:$Id$
// Author: Marek Biskup, Ilka Antcheva   24/07/03

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedPatternSelect
#define ROOT_TGedPatternSelect


#include "TGButton.h"
#include "TGToolTip.h"


class TGedPopup : public TGCompositeFrame {

protected:
   const TGWindow  *fMsgWindow;

public:
   TGedPopup(const TGWindow* p, const TGWindow *m, UInt_t w, UInt_t h,
             UInt_t options = 0, Pixel_t back = GetDefaultFrameBackground());
   ~TGedPopup() override { }

   Bool_t HandleButton(Event_t *event) override;
   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;
   void           PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void           EndPopup();

   ClassDefOverride(TGedPopup,0)  //popup window
};

class TGedPatternFrame : public TGFrame {

protected:
   const TGWindow *fMsgWindow;
   Bool_t          fActive;
   Style_t         fPattern;
   static TGGC    *fgGC;
   TGToolTip      *fTip;         ///< tool tip associated with a button
   char            fTipText[7];

   void    DoRedraw() override;

public:
   TGedPatternFrame(const TGWindow *p, Style_t pattern, Int_t width = 40,
                    Int_t height = 20);
   ~TGedPatternFrame() override { delete fTip; }

   Bool_t  HandleButton(Event_t *event) override;
   Bool_t  HandleCrossing(Event_t *event) override;
   void    DrawBorder() override;

   void            SetActive(Bool_t in) { fActive = in; gClient->NeedRedraw(this); }
   Style_t         GetPattern() const { return fPattern; }
   static void     SetFillStyle(TGGC* gc, Style_t fstyle); //set fill style for given GC

   ClassDefOverride(TGedPatternFrame,0)  //pattern frame
};

class TGedPatternSelector : public TGCompositeFrame {

protected:
   Int_t              fActive;
   const TGWindow    *fMsgWindow;
   TGedPatternFrame  *fCe[27];

public:
   TGedPatternSelector(const TGWindow *p);
   ~TGedPatternSelector() override;

   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;
   void           SetActive(Int_t newat);
   Int_t          GetActive() const { return fActive; }

   ClassDefOverride(TGedPatternSelector,0)  //select pattern frame
};

class TGedPatternPopup : public TGedPopup {

protected:
   Style_t  fCurrentPattern;

public:
   TGedPatternPopup(const TGWindow *p, const TGWindow *m, Style_t pattern);
   ~TGedPatternPopup() override;

   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;

   ClassDefOverride(TGedPatternPopup,0)  // Color selector popup
};

class TGedSelect : public TGCheckButton {

protected:
   TGGC           *fDrawGC;
   TGedPopup      *fPopup;

   void   DoRedraw() override;
   void           DrawTriangle(GContext_t gc, Int_t x, Int_t y);

public:
   TGedSelect(const TGWindow *p, Int_t id);
   ~TGedSelect() override;

   Bool_t HandleButton(Event_t *event) override;

   virtual void   Enable();
   virtual void   Disable();
   virtual void   SetPopup(TGedPopup* p) { fPopup = p; }  // popup will be deleted in destructor.

   ClassDefOverride(TGedSelect,0)  //selection check-button
};

class TGedPatternSelect : public TGedSelect {

protected:
   Style_t      fPattern;

   void DoRedraw() override;

public:
   TGedPatternSelect(const TGWindow *p, Style_t pattern, Int_t id);
   ~TGedPatternSelect() override {}

   void           SetPattern(Style_t pattern, Bool_t emit=kTRUE);
   Style_t        GetPattern() const { return fPattern; }
          TGDimension GetDefaultSize() const override { return TGDimension(55, 21); }
   virtual void   PatternSelected(Style_t pattern = 0)
                  { Emit("PatternSelected(Style_t)", pattern ? pattern : GetPattern()); }  // *SIGNAL*
   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;
   void   SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGedPatternSelect,0)  //pattern selection check-button
};

#endif
