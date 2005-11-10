// @(#)root/gl:$Name:  $:$Id: TGLSAFrame.cxx,v 1.2 2005/08/11 15:28:53 rdm Exp $
// Author:  Richard Maunder  10/08/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLSAFrame.h"
#include "TGLSAViewer.h"

ClassImp(TGLSAFrame)

//______________________________________________________________________________
TGLSAFrame::TGLSAFrame(TGLSAViewer & viewer) :
   TGMainFrame(gClient->GetDefaultRoot()),
   fViewer(viewer)
{
}

//______________________________________________________________________________
TGLSAFrame::~TGLSAFrame()
{
}

//______________________________________________________________________________
Bool_t TGLSAFrame::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   return fViewer.ProcessFrameMessage(msg, parm1, parm2);
}

//______________________________________________________________________________
void TGLSAFrame::CloseWindow()
{
   // Ask our owning viewer to close
   // Has to be defered so that our GUI event thread can process this event
   // and emit signals - otherwise deleted object is called to emit events
   // Not very nice but seems to be only reliable way to close down
   TTimer::SingleShot(50, "TGLSAViewer", &fViewer, "Close()");
}
