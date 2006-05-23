// @(#)root/gui:$Name:  $:$Id: TGColorSelect.h,v 1.10 2006/05/14 10:23:26 brun Exp $
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
// The TGColorFrame is a small frame with border showing a specific     //
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
   const TGWindow *fMsgWindow;   // window handling container messages
   Pixel_t         fPixel;       // color value of this cell
   Bool_t          fActive;      // kTRUE if this color cell is active
   GContext_t      fGrayGC;      // Shadow GC
   Pixel_t         fColor;       // returned color value

   TGColorFrame(const TGColorFrame&);
   TGColorFrame& operator=(const TGColorFrame&);

public:
   TGColorFrame(const TGWindow *p = 0, Pixel_t c = 0, Int_t n = 1);
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
   Int_t            fActive;     // index of active color cell
   const TGWindow  *fMsgWindow;  // window handling container messages
   TGColorFrame    *fCe[16];     // matrix of color cells

   TG16ColorSelector(const TG16ColorSelector&);   
   TG16ColorSelector& operator=(const TG16ColorSelector&);   

public:
   TG16ColorSelector(const TGWindow *p = 0);
   virtual ~TG16ColorSelector();

   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   void    SetActive(Int_t newat);
   Int_t   GetActive() { return fActive; }

   ClassDef(TG16ColorSelector,0)  // 16 color cells
};

//----------------------------------------------------------------------

class TGColorPopup : public TGCompositeFrame {

protected:
   Int_t            fActive;        // active color index
   Int_t            fLaunchDialog;  // flag used for launching color dialog
   const TGWindow  *fMsgWindow;     // window handling container messages
   Pixel_t          fCurrentColor;  // currently selected color value

   TGColorPopup(const TGColorPopup&);
   TGColorPopup& operator=(const TGColorPopup&);

public:
   TGColorPopup(const TGWindow *p = 0, const TGWindow *m = 0, Pixel_t color = 0);
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
   Pixel_t       fColor;         // color value of the button
   TGGC          fDrawGC;        // drawing GC
   TGColorPopup *fColorPopup;    // color popup associated
   TGPosition    fPressPos;      // psotion of frame on button press event

   TGColorSelect(const TGColorSelect&);
   TGColorSelect& operator=(const TGColorSelect&);

   virtual void DoRedraw();

   void DrawTriangle(GContext_t gc, Int_t x, Int_t y);

public:
   TGColorSelect(const TGWindow *p = 0, Pixel_t color = 0, 
                 Int_t id = -1);
   virtual ~TGColorSelect();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   void    SetColor(Pixel_t color, Bool_t emit = kTRUE);
   Pixel_t GetColor() const { return fColor; }
   void    Enable(Bool_t on = kTRUE);  //*TOGGLE* *GETTER=IsEnabled
   void    Disable();

   // dummy methods just to remove from context menu
   void SetDown(Bool_t on = kTRUE, Bool_t emit = kFALSE) { TGButton::SetDown(on, emit); }
   void Rename(const char *title)  { TGTextButton::SetTitle(title); }
   void SetEnabled(Bool_t e = kTRUE) {TGButton::SetEnabled(e); }

   virtual TGDimension GetDefaultSize() const { return TGDimension(43, 21); }
   virtual void SavePrimitive(ofstream &out, Option_t *);

   virtual void ColorSelected(Pixel_t color = 0)
            { Emit("ColorSelected(Pixel_t)", color ? color : GetColor()); }  //*SIGNAL*

   ClassDef(TGColorSelect,0)  // Color selection checkbutton
};

#endif
