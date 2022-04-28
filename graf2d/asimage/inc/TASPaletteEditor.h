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

#include "TAttImage.h"
#include "TGFrame.h"
#include "TLine.h"


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
      void Paint(Option_t *option = "") override;
   };

   class LimitLine : public TLine {
   private:
      TASPaletteEditor  *fGui;
   protected:
      void ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   public:
      LimitLine(Coord_t x, Coord_t y1, Coord_t y2, TASPaletteEditor *gui);
      void Paint(Option_t *option = "") override;
   };

   Double_t              fMinValue;           ///< min value of image
   Double_t              fMaxValue;           ///< max value of image

   TH1D                 *fHisto;              ///< histogram of image pixels
   TRootEmbeddedCanvas  *fPaletteCanvas;      ///< canvas to draw the current palette
   TRootEmbeddedCanvas  *fHistCanvas;         ///< canvas to draw the histogram
   TList                *fPaletteList;        ///< list of palettes for undo and redo
   TImagePalette        *fPalette;            ///< current palette
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

   Bool_t ProcessMessage(Longptr_t msg, Longptr_t param1, Longptr_t param2) override;

   void   UpdateRange();
   void   CloseWindow() override;

   ClassDefOverride(TASPaletteEditor,0)  // GUI to edit a color palette
};

#endif
