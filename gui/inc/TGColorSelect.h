// @(#)root/gui:$Name:  $:$Id: TGColorSelect.h,v 1.1 2002/09/14 00:35:05 rdm Exp $
// Author: Bertrand Bellenot + Fons Rademakers   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGColorSelect
#define ROOT_TGColorSelect

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGColorFrame, TG16ColorSelector, TGColorPopup and TGColorSelect.     //
//                                                                      //
// The TGColorFrame is a small framw with border showing a specific     //
// color.                                                               //
//                                                                      //
// The TG16ColorSelector is a composite frame with 16 TGColorFrames.    //
//                                                                      //
// The TGColorPopup is a popup containing a TG16ColorSelector and a     //
// "More..." button which popups up a TGColorDialog allowing custom     //
// color selection.                                                     //
//                                                                      //
// The TGColorSelect widget is like a checkbutton but instead of the    //
// check mark there is color area with a little down arrow. When        //
// clicked on the arrow the TGColorPopup pops up.                       //
//                                                                      //
// Selecting a color in this widget will generate the event:            //
// kC_COLORSEL, kCOL_SELCHANGED, widget id, pixel.                      //
// and the signal:                                                      //
// ColorSelected(Pixel_t pixel)                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGButton
#include "TGButton.h"
#endif


//----------------------------------------------------------------------

class TGColorFrame : public TGFrame {

protected:
   const TGWindow *fMsgWindow;
   Pixel_t         fPixel;
   Bool_t          fActive;
   GContext_t      fGrayGC;
   Pixel_t         fColor;

public:
   TGColorFrame(const TGWindow *p, Pixel_t c, Int_t n);
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
   Int_t            fActive;
   const TGWindow  *fMsgWindow;
   TGColorFrame    *fCe[16];

public:
   TG16ColorSelector(const TGWindow *p);
   virtual ~TG16ColorSelector();

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   void    SetActive(Int_t newat);
   Int_t   GetActive() { return fActive; }

   ClassDef(TG16ColorSelector,0)  // 16 color cells
};

//----------------------------------------------------------------------

class TGColorPopup : public TGCompositeFrame {

protected:
   Int_t            fActive;
   Int_t            fLaunchDialog;
   const TGWindow  *fMsgWindow;
   Pixel_t          fCurrentColor;

public:
   TGColorPopup(const TGWindow *p, const TGWindow *m, Pixel_t color);
   virtual ~TGColorPopup();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   void    PlacePopup(Int_t x, Int_t y, UInt_t w, UInt_t h);
   void    EndPopup();

   ClassDef(TGColorPopup,0)  // Color selector popup
};

//----------------------------------------------------------------------

class TGColorSelect : public TGCheckButton {

protected:
   Pixel_t       fColor;
   TGGC          fDrawGC;
   TGColorPopup *fColorPopup;

   virtual void DoRedraw();

   void DrawTriangle(GContext_t gc, Int_t x, Int_t y);

public:
   TGColorSelect(const TGWindow *p, Pixel_t color, Int_t id);
   virtual ~TGColorSelect();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   void    SetColor(Pixel_t color);
   Pixel_t GetColor() const { return fColor; }
   void    Enable();
   void    Disable();

   virtual TGDimension GetDefaultSize() const { return TGDimension(43, 21); }

   virtual void ColorSelected() { Emit("ColorSelected(Pixel_t)", GetColor()); }  //*SIGNAL*

   ClassDef(TGColorSelect,0)  // Color selection checkbutton
};

#endif
