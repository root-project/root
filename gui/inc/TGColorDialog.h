// @(#)root/gui:$Name:$:$Id:$
// Author: Bertrand Bellenot + Fons Rademakers   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGColorDialog
#define ROOT_TGColorDialog

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGColorPalette, TGColorPick and TGColorDialog.                       //
//                                                                      //
// The TGColorPalette is a widget showing an matrix of color cells. The //
// colors can be set and selected.                                      //
//                                                                      //
// The TGColorPick is a widget which allows a color to be picked from   //
// HLS space. It consists of two elements: a color map window from      //
// where the user can select the hue and saturation level of a color,   //
// and a slider to select color's lightness.                            //
//                                                                      //
// Selecting a color in these two widgets will generate the event:      //
// kC_COLORSEL, kCOL_CLICK, widget id, 0.                              //
// and the signal:                                                      //
// ColorSelected(ULong_t pixel)                                         //
//                                                                      //
// The TGColorDialog presents a full featured color selection dialog.   //
// It uses 2 TGColorPalette's and the TGColorPick widgets.              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif


class TGTextEntry;
class TGTextBuffer;


//----------------------------------------------------------------------

class TGColorPalette : public TGFrame, public TGWidget {

protected:
   Int_t    fCx;           // x coordinate of currently selected color cell
   Int_t    fCy;           // y coordinate of currently selected color cell
   UInt_t   fCw;           // color cell width
   UInt_t   fCh;           // color cell height
   Int_t    fRows;         // number of color cell rows
   Int_t    fCols;         // number of color cell columns
   ULong_t *fPixels;       // pixel value of colors
   TGGC     fDrawGC;       // graphics context used fro drawing

   virtual void DoRedraw();
   virtual void GotFocus();
   virtual void LostFocus();

   void DrawFocusHilite(Int_t onoff);

public:
   TGColorPalette(const TGWindow *p, Int_t cols, Int_t rows, Int_t id = -1);
   virtual ~TGColorPalette();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);

   virtual TGDimension GetDefaultSize() const
            { return TGDimension((fCw + 5) * fCols, (fCh + 5) * fRows); }

   void    SetColors(ULong_t colors[]);
   void    SetColor(Int_t ix, ULong_t color);
   void    SetCurrentCellColor(ULong_t color);

   void    SetCellSize(Int_t w = 20, Int_t h = 17);

   ULong_t GetColorByIndex(Int_t ix) const { return fPixels[ix]; }
   ULong_t GetCurrentColor() const;

   virtual void ColorSelected()  { Emit("ColorSelected(ULong_t)", GetCurrentColor()); }  //*SIGNAL*

   ClassDef(TGColorPalette,0)  // Color palette widget
};

//----------------------------------------------------------------------

class TGColorPick : public TGFrame, public TGWidget {

private:
   Int_t    fColormap[64][3];    // colormap
   ULong_t  fPixel[64];          // pixel values

protected:
   Pixmap_t     fHSimage;
   Pixmap_t     fLimage;
   Int_t        fNColors;
   Int_t        fClick;
   Int_t        fCx;
   Int_t        fCy;
   Int_t        fCz;
   ULong_t      fCurrentColor;
   Rectangle_t  fColormapRect;
   Rectangle_t  fSliderRect;
   TGGC         fCursorGC;

   virtual void DoRedraw();

   void    DrawHScursor(Int_t onoff);
   void    DrawLcursor(Int_t onoff);
   void    SetHScursor(Int_t x, Int_t y);
   void    SetLcursor(Int_t z);

   void    CreateImages();
   void    InitImages();
   void    SetSliderColor();
   void    UpdateCurrentColor();

   void    AllocColors();
   void    FreeColors();
   void    CreateDitheredImage(Pixmap_t image, Int_t which);

public:
   TGColorPick(const TGWindow *p, Int_t w, Int_t h, Int_t id = -1);
   virtual ~TGColorPick();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);

   void     SetColor(ULong_t color);
   ULong_t  GetCurrentColor() const { return fCurrentColor; }

   virtual void ColorSelected()  { Emit("ColorSelected(ULong_t)", GetCurrentColor()); }  //*SIGNAL*

   ClassDef(TGColorPick,0)  // Color picker widget
};

//----------------------------------------------------------------------

class TGColorDialog : public TGTransientFrame {

protected:
   ULong_t         fCurrentColor;
   Int_t          *fRetc;
   ULong_t        *fRetColor;

   TGColorPalette *fPalette;
   TGColorPalette *fCpalette;
   TGColorPick    *fColors;
   TGFrame        *fSample;
   TGTextEntry    *fRte, *fGte, *fBte, *fHte, *fLte, *fSte;
   TGTextBuffer   *fRtb, *fGtb, *fBtb, *fHtb, *fLtb, *fStb;

   void    UpdateRGBentries(ULong_t *c);
   void    UpdateHLSentries(ULong_t *c);

public:
   TGColorDialog(const TGWindow *p, const TGWindow *m, Int_t *retc = 0,
                 ULong_t *color = 0);
   virtual ~TGColorDialog();

   virtual void CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGColorDialog,0)  // Color selection dialog
};

#endif
