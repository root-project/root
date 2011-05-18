// @(#)root/asimage:$Id$
// Author: Reiner Rohlfs 24/03/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TASPaletteEditor
#define ROOT_TASPaletteEditor


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TASPaletteEditor                                                    //
//                                                                      //
//  This is a GUI window to edit a color palette.                       //
//  It is called by a context menu item of TImage.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TAttImage
#include "TAttImage.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TLine
#include "TLine.h"
#endif


class TVirtualPad;
class TH1D;
class TRootEmbeddedCanvas;
class TGTextButton;
class TGCheckButton;
class TGComboBox;
class TGRadioButton;


class TASPaletteEditor : public TPaletteEditor, public TGMainFrame {

protected:
   class PaintPalette : public TObject {
   protected :
      TImagePalette  **fPalette;
      TAttImage       *fAttImage;
   public:
      PaintPalette(TImagePalette **palette, TAttImage *attImage)
         { fPalette = palette; fAttImage = attImage; }
      void Paint(Option_t *option);
   };

   class LimitLine : public TLine {
   private:
      TASPaletteEditor  *fGui;
   protected:
      virtual void ExecuteEvent(Int_t event, Int_t px, Int_t py);
   public:
      LimitLine(Coord_t x, Coord_t y1, Coord_t y2, TASPaletteEditor *gui);
      void Paint(Option_t *option);
   };

   Double_t              fMinValue;           // min value of image
   Double_t              fMaxValue;           // max value of image

   TH1D                 *fHisto;              // hitogram of image pixels
   TRootEmbeddedCanvas  *fPaletteCanvas;      // canvas to draw the current palette
   TRootEmbeddedCanvas  *fHistCanvas;         // canvas to draw the histogram
   TList                *fPaletteList;        // list of palettes for undo and redo
   TImagePalette        *fPalette;            // current palette
   TVirtualPad          *fImagePad;
   PaintPalette         *fPaintPalette;
   LimitLine            *fLimitLine[2];

   TGTextButton         *fUnDoButton;
   TGTextButton         *fReDoButton;

   TGCheckButton        *fAutoUpdate;
   TGCheckButton        *fStepButton;
   TGRadioButton        *fRamps[3];
   Int_t                 fRampFactor;

   TGComboBox           *fComboBox;

   void  InsertNewPalette(TImagePalette *newPalette);

   void  Save();
   void  Open();

   void  LogPalette();
   void  ExpPalette();
   void  LinPalette();
   void  InvertPalette();
   void  NewPalette(Long_t id);
   void  SetStep();
   void  SetRamp(Long_t ramp);

   void  UpdateScreen(Bool_t histoUpdate);

public:
   TASPaletteEditor(TAttImage *attImage, UInt_t w, UInt_t h);
   virtual ~TASPaletteEditor();

   Bool_t ProcessMessage(Long_t msg, Long_t param1, Long_t param2);

   void   UpdateRange();
   void   CloseWindow();

   ClassDef(TASPaletteEditor,0)  // GUI to edit a color palette
};

#endif
