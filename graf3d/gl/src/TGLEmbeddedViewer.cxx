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

//==============================================================================
// TGLEmbeddedViewer
//==============================================================================

//______________________________________________________________________________
//
// Minimal GL-viewer that can be embedded in a standard ROOT frames.

ClassImp(TGLEmbeddedViewer);

//______________________________________________________________________________
TGLEmbeddedViewer::TGLEmbeddedViewer(const TGWindow *parent, TVirtualPad *pad) :
   TGLViewer(pad, 0, 0, 400, 300),
   fFrame(0)
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

   delete fFrame;
   fGLWidget = 0;
}

//______________________________________________________________________________
void TGLEmbeddedViewer::CreateFrames()
{
   // Internal frames creation.

   fGLWidget = new TGLWidget(*fFrame, kTRUE, 10, 10, 0);
   // Direct events from the TGWindow directly to the base viewer

   fEventHandler = new TGLEventHandler("Default", fGLWidget, this);
   fGLWidget->SetEventHandler(fEventHandler);

   fFrame->AddFrame(fGLWidget, new TGLayoutHints(kLHintsExpandX |
                    kLHintsExpandY, 2, 2, 2, 2));
}

