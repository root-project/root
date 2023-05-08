// @(#)root/gui:$Id$
// Author: Bertrand Bellenot + Fons Rademakers   22/08/02
// Author: Ilka Antcheva (color wheel support)   16/03/07

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGColorDialog
#define ROOT_TGColorDialog


#include "TGFrame.h"
#include "TGWidget.h"
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

   TGColorPalette(const TGColorPalette&) = delete;
   TGColorPalette& operator=(const TGColorPalette&) = delete;

protected:
   Int_t    fCx;           ///< x coordinate of currently selected color cell
   Int_t    fCy;           ///< y coordinate of currently selected color cell
   UInt_t   fCw;           ///< color cell width
   UInt_t   fCh;           ///< color cell height
   Int_t    fRows;         ///< number of color cell rows
   Int_t    fCols;         ///< number of color cell columns
   Pixel_t *fPixels;       ///< pixel value of colors
   TGGC     fDrawGC;       ///< graphics context used for drawing

   void DoRedraw() override;
   virtual void GotFocus();
   virtual void LostFocus();

   void DrawFocusHilite(Int_t onoff);

public:
   TGColorPalette(const TGWindow *p = nullptr, Int_t cols = 8, Int_t rows = 8, Int_t id = -1);
   virtual ~TGColorPalette();

   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override;
   Bool_t HandleKey(Event_t *event) override;

   TGDimension GetDefaultSize() const override
      { return TGDimension((fCw + 5) * fCols, (fCh + 5) * fRows); }

   void    SetColors(Pixel_t colors[]);
   void    SetColor(Int_t ix, Pixel_t color);
   void    SetCurrentCellColor(Pixel_t color);

   void    SetCellSize(Int_t w = 20, Int_t h = 17);

   Pixel_t GetColorByIndex(Int_t ix) const { return fPixels[ix]; }
   Pixel_t GetCurrentColor() const;

   virtual void ColorSelected(Pixel_t col = 0)
            { Emit("ColorSelected(Pixel_t)", col ? col : GetCurrentColor()); }  //*SIGNAL*

   ClassDefOverride(TGColorPalette,0)  // Color palette widget
};

//----------------------------------------------------------------------

class TGColorPick : public TGFrame, public TGWidget {

private:
   Int_t    fColormap[64][3];    // colormap
   Pixel_t  fPixel[64];          // pixel values

protected:
   Pixmap_t     fHSimage;        ///< hue / saturation colormap pixmap
   Pixmap_t     fLimage;         ///< color lightness slider pixmap
   Int_t        fNColors;        ///< number of color samples
   Int_t        fClick;          ///< mouse click location (kCLICK_NONE, kCLICK_HS, kCLICK_L)
   Int_t        fCx;             ///< x position in hs colormap
   Int_t        fCy;             ///< y position in hs colormap
   Int_t        fCz;             ///< position in lightness slider
   Pixel_t      fCurrentColor;   ///< currently selected color value
   Rectangle_t  fColormapRect;   ///< hue / saturation colormap rectangle
   Rectangle_t  fSliderRect;     ///< color lightness slider rectangle
   TGGC         fCursorGC;       ///< color lightness slider cursor GC

   void    DoRedraw() override;

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
   TGColorPick(const TGWindow *p = nullptr, Int_t w = 1, Int_t h = 1, Int_t id = -1);
   virtual ~TGColorPick();

   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override;

   void     SetColor(Pixel_t color);
   Pixel_t  GetCurrentColor() const { return fCurrentColor; }

   virtual  void ColorSelected(Pixel_t col = 0)
            { Emit("ColorSelected(Pixel_t)", col ? col : GetCurrentColor()); }  //*SIGNAL*

   ClassDefOverride(TGColorPick,0)  // Color picker widget
};

//----------------------------------------------------------------------

class TGColorDialog : public TGTransientFrame {

private:

   TGColorDialog(const TGColorDialog&) = delete;
   TGColorDialog& operator=(const TGColorDialog&) = delete;

protected:
   Pixel_t         fCurrentColor;   ///< currently selected color
   Pixel_t         fInitColor;      ///< initially set color
   Int_t          *fRetc;           ///< return code (kMBOk, kMBCancel)
   Pixel_t        *fRetColor;       ///< return color
   TColor         *fRetTColor;      ///< return TColor, needed for changed alpha

   TGColorPalette *fPalette;        ///< color palette
   TGColorPalette *fCpalette;       ///< color palette
   TGColorPick    *fColors;         ///< color pick widget
   TGFrame        *fSample;         ///< color sample frame
   TGFrame        *fSampleOld;      ///< color sample frame
   TGTextEntry    *fRte, *fGte, *fBte, *fHte, *fLte, *fSte, *fAle; ///< RGB/HLS text entries
   TGTextBuffer   *fRtb, *fGtb, *fBtb, *fHtb, *fLtb, *fStb, *fAlb; ///< RGB/HLS associated buffers
   Bool_t          fWaitFor;         ///< call WaitFor method in constructor

   TGTab               *fTab;        ///< tab widget holding the color selectors
   TRootEmbeddedCanvas *fEcanvas;    ///< embedded canvas holding the color wheel
   TColorWheel         *fColorWheel; ///< color wheel
   TGLabel             *fColorInfo;  ///< color info
   TGTextButton        *fPreview;    ///< preview button;

   void   UpdateRGBentries(Pixel_t *c);
   void   UpdateHLSentries(Pixel_t *c);
   void   UpdateAlpha(Pixel_t *c);
   void   CloseWindow() override;
   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;

public:
   TGColorDialog(const TGWindow *p = nullptr, const TGWindow *m = nullptr, Int_t *retc = nullptr,
                 Pixel_t *color = nullptr, Bool_t wait = kTRUE);
   virtual ~TGColorDialog();

   TGColorPalette *GetPalette() const { return fPalette; }
   TGColorPalette *GetCustomPalette() const { return fCpalette; }

   virtual void ColorSelected(Pixel_t); //*SIGNAL*
   virtual void AlphaColorSelected(ULongptr_t); //*SIGNAL*
           void DoPreview();
   virtual void SetCurrentColor(Pixel_t col);
           void SetColorInfo(Int_t event, Int_t px, Int_t py, TObject *selected);

   ClassDefOverride(TGColorDialog,0)  // Color selection dialog
};

#endif
