// @(#)root/ged:$Name:  $:$Id: TGedEditor.cxx,v 1.0 2003/12/02 13:41:59 rdm Exp $
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
#include "TGedToolBox.h"
#include "TCanvas.h"
#include "TGedPropertyFrame.h"

ClassImp(TGedEditor)

//______________________________________________________________________________
TGedEditor::TGedEditor(TCanvas* canvas) :
   TGMainFrame(gClient->GetRoot(), 110, 20)
{

   Build();
   if (canvas)
      ConnectToCanvas(canvas);
      
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();
}

//______________________________________________________________________________
void TGedEditor::Build()
{
   fToolBox = new TGedToolBox(this, 110, 20, 0);
   AddFrame(fToolBox, 
            new TGLayoutHints(kLHintsTop |  kLHintsExpandX , 0, 0, 2, 2));
   fPropertiesFrame = new TGedPropertyFrame(this);
   AddFrame(fPropertiesFrame,
             new TGLayoutHints(kLHintsTop |  kLHintsExpandX , 0, 0, 2, 2));
}

//______________________________________________________________________________
void TGedEditor::ConnectToCanvas(TCanvas *c)
{
   fPropertiesFrame->ConnectToCanvas(c);
}

//______________________________________________________________________________
TGedEditor::~TGedEditor()
{
   Cleanup();
}
