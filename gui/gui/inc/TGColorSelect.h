// @(#)root/gui:$Id$
// Author: Bertrand Bellenot + Fons Rademakers   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGColorSelect
#define ROOT_TGColorSelect


#include "TGFrame.h"
#include "TGButton.h"


//----------------------------------------------------------------------

class TGColorFrame : public TGFrame {

protected:
   const TGWindow *fMsgWindow;   ///< window handling container messages
   Pixel_t         fPixel;       ///< color value of this cell
   Bool_t          fActive;      ///< kTRUE if this color cell is active
   GContext_t      fGrayGC;      ///< Shadow GC
   Pixel_t         fColor;       ///< returned color value

private:
   TGColorFrame(const TGColorFrame&) = delete;
   TGColorFrame& operator=(const TGColorFrame&) = delete;

public:
   TGColorFrame(const TGWindow *p = nullptr, Pixel_t c = 0, Int_t n = 1);
   virtual ~TGColorFrame() { }

   virtual Bool_t  HandleButton(Event_t *event);
   virtual void    DrawBorder();

   void     SetActive(Bool_t in) { fActive = in; gClient->NeedRedraw(this); }
   Pixel_t  GetColor() const { return fColor; }

   ClassDef(TGColorFrame,0)  // Frame for color cell
};

//----------------------------------------------------------------------

class TG16ColorSelector : public TGCompositeFrame {

protected:
   Int_t            fActive;     ///< index of active color cell
   const TGWindow  *fMsgWindow;  ///< window handling container messages
   TGColorFrame    *fCe[16];     ///< matrix of color cells

private:
   TG16ColorSelector(const TG16ColorSelector&) = delete;
   TG16ColorSelector& operator=(const TG16ColorSelector&) = delete;

public:
   TG16ColorSelector(const TGWindow *p = nullptr);
   virtual ~TG16ColorSelector();

   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);

   void    SetActive(Int_t newat);
   Int_t   GetActive() { return fActive; }

   ClassDef(TG16ColorSelector,0)  // 16 color cells
};

//----------------------------------------------------------------------

class TGColorPopup : public TGCompositeFrame {

protected:
   Int_t            fActive;        ///< active color index
   Int_t            fLaunchDialog;  ///< flag used for launching color dialog
   const TGWindow  *fMsgWindow;     ///< window handling container messages
   Pixel_t          fCurrentColor;  ///< currently selected color value

private:
   TGColorPopup(const TGColorPopup&) = delete;
   TGColorPopup& operator=(const TGColorPopup&) = delete;

public:
   TGColorPopup(const TGWindow *p = nullptr, const TGWindow *m = nullptr, Pixel_t color = 0);
   virtual ~TGColorPopup();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);

   void    PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void    EndPopup();
   void    PreviewColor(Pixel_t color);
   void    PreviewAlphaColor(ULongptr_t color);

   ClassDef(TGColorPopup,0)  // Color selector popup
};

//----------------------------------------------------------------------

class TGColorSelect : public TGCheckButton {

protected:
   Pixel_t       fColor;         ///< color value of the button
   TGGC          fDrawGC;        ///< drawing GC
   TGColorPopup *fColorPopup;    ///< color popup associated
   TGPosition    fPressPos;      ///< position of frame on button press event

   virtual void DoRedraw();

   void DrawTriangle(GContext_t gc, Int_t x, Int_t y);

private:
   TGColorSelect(const TGColorSelect&) = delete;
   TGColorSelect& operator=(const TGColorSelect&) = delete;

public:
   TGColorSelect(const TGWindow *p = nullptr, Pixel_t color = 0,
                 Int_t id = -1);
   virtual ~TGColorSelect();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2);

   void    SetColor(Pixel_t color, Bool_t emit = kTRUE);
   void    SetAlphaColor(ULong_t color, Bool_t emit = kTRUE);
   Pixel_t GetColor() const { return fColor; }
   void    Enable(Bool_t on = kTRUE);  //*TOGGLE* *GETTER=IsEnabled
   void    Disable();

   // dummy methods just to remove from context menu
   void SetDown(Bool_t on = kTRUE, Bool_t emit = kFALSE) { TGButton::SetDown(on, emit); }
   void Rename(const char *title)  { TGTextButton::SetTitle(title); }
   void SetEnabled(Bool_t e = kTRUE) { TGButton::SetEnabled(e); }

   virtual TGDimension GetDefaultSize() const { return TGDimension(43, 21); }
   virtual void SavePrimitive(std::ostream &out, Option_t * = "");

   virtual void ColorSelected(Pixel_t color = 0)
            { Emit("ColorSelected(Pixel_t)", color ? color : GetColor()); }  //*SIGNAL*
   virtual void AlphaColorSelected(ULong_t colptr = 0)
            { Emit("AlphaColorSelected(ULong_t)", colptr); }  //*SIGNAL*

   ClassDef(TGColorSelect,0)  // Color selection checkbutton
};

#endif
