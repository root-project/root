// @(#)root/gl:$Id$
// Author: Bertrand Bellenot 2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLOverlayButton
#define ROOT_TGLOverlayButton

#include "TGLOverlay.h"
#include "TGLFontManager.h"
#include "TGLViewerBase.h"
#include "TQObject.h"

class TString;

class TGLOverlayButton : public TGLOverlayElement,
                         public TQObject
{

private:
   TGLOverlayButton(const TGLOverlayButton&);            // Not implemented
   TGLOverlayButton& operator=(const TGLOverlayButton&); // Not implemented

protected:

   TString           fText;         // button text
   Int_t             fActiveID;     // active item identifier
   Pixel_t           fBackColor;    // button background color
   Pixel_t           fTextColor;    // text color
   Float_t           fNormAlpha;    // button alpha value (transparency) in normal state
   Float_t           fHighAlpha;    // button alpha value (transparency) in highlight state

   Float_t           fPosX;         // button x position
   Float_t           fPosY;         // button y position
   Float_t           fWidth;        // button width
   Float_t           fHeight;       // button height

   mutable TGLFont   fFont;         // font used to render text

public:
   TGLOverlayButton(TGLViewerBase *parent, const char *text, Float_t posx,
                    Float_t posy, Float_t width, Float_t height);
   virtual ~TGLOverlayButton() { }

   virtual Bool_t       MouseEnter(TGLOvlSelectRecord& selRec);
   virtual Bool_t       Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec, Event_t* event);
   virtual void         MouseLeave();

   virtual void         Render(TGLRnrCtx& rnrCtx);
   virtual void         ResetState() { fActiveID = -1; }

   virtual const char  *GetText() const { return fText.Data(); }
   virtual Pixel_t      GetBackColor() const { return fBackColor; }
   virtual Pixel_t      GetTextColor() const { return fTextColor; }
   virtual void         SetText(const char *text) { fText = text; }
   virtual void         SetPosition(Float_t x, Float_t y) { fPosX = x; fPosY = y; }
   virtual void         SetSize(Float_t w, Float_t h) { fWidth = w; fHeight = h; }
   virtual void         SetAlphaValues(Float_t norm, Float_t high) { fNormAlpha = norm; fHighAlpha = high; }
   virtual void         SetBackColor(Pixel_t col) { fBackColor = col; }
   virtual void         SetTextColor(Pixel_t col) { fTextColor = col; }

   virtual void         Clicked(TGLViewerBase *viewer); // *SIGNAL*

   ClassDef(TGLOverlayButton, 0); // GL-overlay button.
};

#endif

