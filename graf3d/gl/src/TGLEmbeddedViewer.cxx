// @(#)root/gl:$Id$
// Author: Bertrand Bellenot 23/01/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGFrame.h"
#include "TGLayout.h"
#include "TGLWidget.h"
#include "TGLSAFrame.h"
#include "TString.h"
#include "TGLPShapeObj.h"
#include "TGLPhysicalShape.h"
#include "TGLLogicalShape.h"
#include "TGLEmbeddedViewer.h"
#include "TGLEventHandler.h"

ClassImp(TGLEmbeddedViewer)

//______________________________________________________________________________
TGLEmbeddedViewer::TGLEmbeddedViewer(const TGWindow *parent, TVirtualPad *pad) :
   TGLViewer(pad, 0, 0, 400, 300),
   fFrame(0),
   fGLArea(0)
{
   // Default constructor;

   fFrame = new TGCompositeFrame(parent);

   CreateFrames();

   fFrame->MapSubwindows();
   fFrame->Resize(fFrame->GetDefaultSize());
   fFrame->Resize(400, 300);
}

//______________________________________________________________________________
TGLEmbeddedViewer::~TGLEmbeddedViewer()
{
   // Destroy standalone viewer object.

//   delete fGLArea;
   if (fEventHandler) {
      fGLWindow->SetEventHandler(0);
      delete fEventHandler;
   }
   delete fFrame;
}

//______________________________________________________________________________
void TGLEmbeddedViewer::CreateFrames()
{
   // Internal frames creation.

   fGLWindow = new TGLWidget(*fFrame, kTRUE, 10, 10, 0);
   // Direct events from the TGWindow directly to the base viewer

   fEventHandler = new TGLEventHandler("Default", fGLWindow, this);
   fGLWindow->SetEventHandler(fEventHandler);

   fFrame->AddFrame(fGLWindow, new TGLayoutHints(kLHintsExpandX |
                    kLHintsExpandY, 2, 2, 2, 2));
}

