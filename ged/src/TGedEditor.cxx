// @(#)root/ged:$Name:  $:$Id: TGedEditor.cxx,v 1.4 2004/03/23 15:22:24 brun Exp $
// Author: Marek Biskup, Ilka Antcheva 02/08/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedEditor                                                           //
//                                                                      //
// Editor is a composite frame that contains TGedToolBox and            //
// TGedAttFrames. It is connected to a Canvas and listens for           //
// selected objects                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGedEditor.h"
#include "TCanvas.h"
#include "TGTab.h"
#include "TGedPropertyFrame.h"
#include "TGLabel.h"

ClassImp(TGedEditor)

//______________________________________________________________________________
TGedEditor::TGedEditor(TCanvas* canvas) :
   TGMainFrame(gClient->GetRoot(), 110, 20)
{

   Build();

   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();

   if (canvas)
      ConnectToCanvas(canvas);

   // to make the height of the global pad editor equal to 
   // the height of the embedded one
   Resize(GetWidth(), canvas->GetWh());
}

//______________________________________________________________________________
void TGedEditor::Build()
{

   fPropertiesFrame = new TGedPropertyFrame(this);
   AddFrame(fPropertiesFrame,
            new TGLayoutHints(kLHintsTop |  kLHintsExpandX , 0, 0, 2, 2));

}

//______________________________________________________________________________
void TGedEditor::CloseWindow()
{
   // When closed via WM close button, just unmap (i.e. hide) editor
   // for later use.

   Hide();
}

//______________________________________________________________________________
void TGedEditor::ConnectToCanvas(TCanvas *c)
{
   fPropertiesFrame->ConnectToCanvas(c);
}

//______________________________________________________________________________
void TGedEditor::Show()
{
   // Show editor.

   MapWindow();
}

//______________________________________________________________________________
void TGedEditor::Hide()
{
   // Hide editor.

   UnmapWindow();
}

//______________________________________________________________________________
TGedEditor::~TGedEditor()
{
   Cleanup();
}
