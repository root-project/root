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
   virtual ~TGedPopup() { }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
   void           PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void           EndPopup();

   ClassDef(TGedPopup,0)  //popup window
};

class TGedPatternFrame : public TGFrame {

protected:
   const TGWindow *fMsgWindow;
   Bool_t          fActive;
   Style_t         fPattern;
   static TGGC    *fgGC;
   TGToolTip      *fTip;         ///< tool tip associated with a button
   char            fTipText[7];

   virtual void    DoRedraw();

public:
   TGedPatternFrame(const TGWindow *p, Style_t pattern, Int_t width = 40,
                    Int_t height = 20);
   virtual ~TGedPatternFrame() { delete fTip; }

   virtual Bool_t  HandleButton(Event_t *event);
   virtual Bool_t  HandleCrossing(Event_t *event);
   virtual void    DrawBorder();

   void            SetActive(Bool_t in) { fActive = in; gClient->NeedRedraw(this); }
   Style_t         GetPattern() const { return fPattern; }
   static void     SetFillStyle(TGGC* gc, Style_t fstyle); //set fill style for given GC

   ClassDef(TGedPatternFrame,0)  //pattern frame
};

class TGedPatternSelector : public TGCompositeFrame {

protected:
   Int_t              fActive;
   const TGWindow    *fMsgWindow;
   TGedPatternFrame  *fCe[27];

public:
   TGedPatternSelector(const TGWindow *p);
   virtual ~TGedPatternSelector();

   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
   void           SetActive(Int_t newat);
   Int_t          GetActive() const { return fActive; }

   ClassDef(TGedPatternSelector,0)  //select pattern frame
};

class TGedPatternPopup : public TGedPopup {

protected:
   Style_t  fCurrentPattern;

public:
   TGedPatternPopup(const TGWindow *p, const TGWindow *m, Style_t pattern);
   virtual ~TGedPatternPopup();

   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);

   ClassDef(TGedPatternPopup,0)  // Color selector popup
};

class TGedSelect : public TGCheckButton {

protected:
   TGGC           *fDrawGC;
   TGedPopup      *fPopup;

   virtual void   DoRedraw();
   void           DrawTriangle(GContext_t gc, Int_t x, Int_t y);

public:
   TGedSelect(const TGWindow *p, Int_t id);
   virtual ~TGedSelect();

   virtual Bool_t HandleButton(Event_t *event);

   virtual void   Enable();
   virtual void   Disable();
   virtual void   SetPopup(TGedPopup* p) { fPopup = p; }  // popup will be deleted in destructor.

   ClassDef(TGedSelect,0)  //selection check-button
};

class TGedPatternSelect : public TGedSelect {

protected:
   Style_t      fPattern;

   virtual void DoRedraw();

public:
   TGedPatternSelect(const TGWindow *p, Style_t pattern, Int_t id);
   virtual ~TGedPatternSelect() {}

   void           SetPattern(Style_t pattern, Bool_t emit=kTRUE);
   Style_t        GetPattern() const { return fPattern; }
   virtual        TGDimension GetDefaultSize() const { return TGDimension(55, 21); }
   virtual void   PatternSelected(Style_t pattern = 0)
                  { Emit("PatternSelected(Style_t)", pattern ? pattern : GetPattern()); }  // *SIGNAL*
   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);
   virtual void   SavePrimitive(std::ostream &out, Option_t * = "");

   ClassDef(TGedPatternSelect,0)  //pattern selection check-button
};

#endif
