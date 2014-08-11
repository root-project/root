// @(#)root/gui:$Id$
// Author: Bertrand Bellenot + Fons Rademakers   22/08/02
// Author: Ilka Antcheva (color wheel support)   16/03/07

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
// kC_COLORSEL, kCOL_CLICK, widget id, 0.                               //
// and the signal:                                                      //
// ColorSelected(Pixel_t color)                                         //
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
#include "TColor.h"


class TGTextEntry;
class TGTextBuffer;
class TGTab;
class TRootEmbeddedCanvas;
class TColorWheel;
class TGLabel;
class TGTextButton;

//----------------------------------------------------------------------

class TGColorPalette : public TGFrame, public TGWidget {

private:

   TGColorPalette(const TGColorPalette&); // Not implemented
   TGColorPalette& operator=(const TGColorPalette&); // Not implemented

protected:
   Int_t    fCx;           // x coordinate of currently selected color cell
   Int_t    fCy;           // y coordinate of currently selected color cell
   UInt_t   fCw;           // color cell width
   UInt_t   fCh;           // color cell height
   Int_t    fRows;         // number of color cell rows
   Int_t    fCols;         // number of color cell columns
   Pixel_t *fPixels;       // pixel value of colors
   TGGC     fDrawGC;       // graphics context used for drawing

   virtual void DoRedraw();
   virtual void GotFocus();
   virtual void LostFocus();

   void DrawFocusHilite(Int_t onoff);

public:
   TGColorPalette(const TGWindow *p = 0, Int_t cols = 8, Int_t rows = 8, Int_t id = -1);
   virtual ~TGColorPalette();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);

   virtual TGDimension GetDefaultSize() const
            { return TGDimension((fCw + 5) * fCols, (fCh + 5) * fRows); }

   void    SetColors(Pixel_t colors[]);
   void    SetColor(Int_t ix, Pixel_t color);
   void    SetCurrentCellColor(Pixel_t color);

   void    SetCellSize(Int_t w = 20, Int_t h = 17);

   Pixel_t GetColorByIndex(Int_t ix) const { return fPixels[ix]; }
   Pixel_t GetCurrentColor() const;

   virtual void ColorSelected(Pixel_t col = 0)
            { Emit("ColorSelected(Pixel_t)", col ? col : GetCurrentColor()); }  //*SIGNAL*

   ClassDef(TGColorPalette,0)  // Color palette widget
};

//----------------------------------------------------------------------

class TGColorPick : public TGFrame, public TGWidget {

private:
   Int_t    fColormap[64][3];    // colormap
   Pixel_t  fPixel[64];          // pixel values

protected:
   Pixmap_t     fHSimage;        // hue / saturation colormap pixmap
   Pixmap_t     fLimage;         // color lightness slider pixmap
   Int_t        fNColors;        // number of color samples
   Int_t        fClick;          // mouse click location (kCLICK_NONE, kCLICK_HS, kCLICK_L)
   Int_t        fCx;             // x position in hs colormap
   Int_t        fCy;             // y position in hs colormap
   Int_t        fCz;             // position in lightness slider
   Pixel_t      fCurrentColor;   // currently selected color value
   Rectangle_t  fColormapRect;   // hue / saturation colormap rectangle
   Rectangle_t  fSliderRect;     // color lightness slider rectangle
   TGGC         fCursorGC;       // color lightness slider cursor GC

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
   TGColorPick(const TGWindow *p = 0, Int_t w = 1, Int_t h = 1, Int_t id = -1);
   virtual ~TGColorPick();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);

   void     SetColor(Pixel_t color);
   Pixel_t  GetCurrentColor() const { return fCurrentColor; }

   virtual  void ColorSelected(Pixel_t col = 0)
            { Emit("ColorSelected(Pixel_t)", col ? col : GetCurrentColor()); }  //*SIGNAL*

   ClassDef(TGColorPick,0)  // Color picker widget
};

//----------------------------------------------------------------------

class TGColorDialog : public TGTransientFrame {

private:

   TGColorDialog(const TGColorDialog&); // Not implemented
   TGColorDialog& operator=(const TGColorDialog&); // Not implemented

protected:
   Pixel_t         fCurrentColor;   // currently selected color
   Pixel_t         fInitColor;      // initially set color
   Int_t          *fRetc;           // return code (kMBOk, kMBCancel)
   Pixel_t        *fRetColor;       // return color
   TColor         *fRetTColor;      // return TColor, needed for changed alpha

   TGColorPalette *fPalette;        // color palette
   TGColorPalette *fCpalette;       // color palette
   TGColorPick    *fColors;         // color pick widget
   TGFrame        *fSample;         // color sample frame
   TGFrame        *fSampleOld;      // color sample frame
   TGTextEntry    *fRte, *fGte, *fBte, *fHte, *fLte, *fSte, *fAle; // RGB/HLS text entries
   TGTextBuffer   *fRtb, *fGtb, *fBtb, *fHtb, *fLtb, *fStb, *fAlb; // RGB/HLS associated buffers
   Bool_t          fWaitFor;         // call WaitFor method in constructor

   TGTab               *fTab;        //tab widget holding the color selectors
   TRootEmbeddedCanvas *fEcanvas;    //embedded canvas holding the color wheel
   TColorWheel         *fColorWheel; //color wheel
   TGLabel             *fColorInfo;  //color info
   TGTextButton        *fPreview;    //preview button;

   void           UpdateRGBentries(Pixel_t *c);
   void           UpdateHLSentries(Pixel_t *c);
   void           UpdateAlpha(Pixel_t *c);
   virtual void   CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

public:
   TGColorDialog(const TGWindow *p = 0, const TGWindow *m = 0, Int_t *retc = 0,
                 Pixel_t *color = 0, Bool_t wait = kTRUE);
   virtual ~TGColorDialog();

   TGColorPalette *GetPalette() const { return fPalette; }
   TGColorPalette *GetCustomPalette() const { return fCpalette; }

   virtual void ColorSelected(Pixel_t); //*SIGNAL*
   virtual void AlphaColorSelected(ULong_t); //*SIGNAL*
           void DoPreview();
   virtual void SetCurrentColor(Pixel_t col);
           void SetColorInfo(Int_t event, Int_t px, Int_t py, TObject *selected);

   ClassDef(TGColorDialog,0)  // Color selection dialog
};

#endif
