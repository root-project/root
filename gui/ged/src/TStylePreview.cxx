// @(#)root/ged:$Id$
// Author: Denis Favre-Miville   08/09/05

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TStylePreview                                                       //
//                                                                      //
//  This class may be used to preview the result of applying a style    //
//       to a canvas. The result is shown on a clone of the object,     //
//       in a different shown over the initial canvas.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TStylePreview.h"
#include "TStyleManager.h"

#include <TCanvas.h>
#include <TRootEmbeddedCanvas.h>
#include <TStyle.h>
#include <TROOT.h>

ClassImp(TStylePreview)

//______________________________________________________________________________
TStylePreview::TStylePreview(const TGWindow *p, TStyle *style,
                              TVirtualPad *currentPad)
                     : TGTransientFrame(0, p)
{
   //  Constructor. Create a new window and draw a clone of
   // currentPad->GetCanvas() in it, using the style 'style'.
   //  Thanks to that method, one can have a preview of any
   // style with any object.

   fPad = 0;

   // Create the main window.
   SetWindowName("Style Manager's Preview");
   SetCleanup(kNoCleanup);
   DontCallClose();

   // Create the trash lists to have an effective deletion of every object.
   fTrashListLayout = new TList();

   // Create the layouts and add them to the layout trash list.
   TGLayoutHints *layoutXY = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   fTrashListLayout->Add(layoutXY);

   // Create a canvas for the preview.
   fEcan = new TRootEmbeddedCanvas("TSMPreviewCanvas", this, 10, 10);
   AddFrame(fEcan, layoutXY);

   // Draw the preview.
   Update(style, currentPad);

   // Map main frame.
   MapTheWindow();

   // No modifications allowed in the preview.
   fEcan->GetCanvas()->SetEditable(kFALSE);
   fEcan->GetCanvas()->SetBit(kNoContextMenu);
}

//______________________________________________________________________________
TStylePreview::~TStylePreview()
{
   // Destructor.

   // Delete all the widgets created in this class.
   delete fEcan;

   // Delete all the layouts.
   TObject *obj1;
   TObject *obj2;
   obj1 = fTrashListLayout->First();
   while (obj1) {
      obj2 = fTrashListLayout->After(obj1);
      fTrashListLayout->Remove(obj1);
      delete obj1;
      obj1 = obj2;
   }
   delete fTrashListLayout;
}

//______________________________________________________________________________
void TStylePreview::Update(TStyle *style, TVirtualPad *pad)
{
   // Update the preview with possibly another style and
   // another object than previously.

   TCanvas *c;
   if (pad != fPad) {
      delete fEcan->GetCanvas();
      fEcan->AdoptCanvas(new TCanvas("TSMPreviewCanvas", 10, 10,
                                       fEcan->GetCanvasWindowId()));
      c = fEcan->GetCanvas();
      gROOT->SetSelectedPad(c);
      if (pad->GetCanvas())
         pad->GetCanvas()->DrawClonePad();
      gROOT->SetSelectedPad(pad);
      fPad = pad;
   }

   // Apply the 'style' to the clone of 'pad'.
   c = fEcan->GetCanvas();
   TStyle *tmpStyle = gStyle;
   gStyle = style;
   c->UseCurrentStyle();
   gStyle = tmpStyle;
   c->Modified();
   c->Update();
}

//______________________________________________________________________________
void TStylePreview::MapTheWindow()
{
   // Initialize the layout algorithm.

   MapSubwindows();
   TCanvas *c = fPad->GetCanvas();
   if (c) {
      UInt_t w = c->GetWw() + 4; //4 pixels of borders
      UInt_t h = c->GetWh() + 4; //4 pixels of borders
      UInt_t x = (UInt_t) c->GetWindowTopX() + 60;
      UInt_t y = (UInt_t) c->GetWindowTopY() + 100;

      MoveResize(x, y, w, h);
      SetWMPosition(x, y);
   }
   MapWindow();
}

//______________________________________________________________________________
TCanvas *TStylePreview::GetMainCanvas()
{
   // Return pointer to the selected canvas.

   return fEcan->GetCanvas();
}
