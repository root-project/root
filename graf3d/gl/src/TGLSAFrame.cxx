// @(#)root/gl:$Id$
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

/** \class TGLSAFrame
\ingroup opengl
Standalone GL Viewer GUI main frame. Is aggregated in TGLSAViewer -
top level standalone viewer object.
*/

ClassImp(TGLSAFrame);

////////////////////////////////////////////////////////////////////////////////
/// Construct GUI frame, bound to passed 'viewer'

TGLSAFrame::TGLSAFrame(TGLSAViewer & viewer) :
   TGMainFrame(gClient->GetRoot()),
   fViewer(viewer)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Construct GUI frame, bound to passed 'viewer'

TGLSAFrame::TGLSAFrame(const TGWindow* parent, TGLSAViewer & viewer) :
   TGMainFrame(parent),
   fViewer(viewer)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy the GUI frame

TGLSAFrame::~TGLSAFrame()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Process GUI message - deferred back up to TGLSAViewer::ProcessFrameMessage()

Bool_t TGLSAFrame::ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2)
{
   return fViewer.ProcessFrameMessage(msg, parm1, parm2);
}

////////////////////////////////////////////////////////////////////////////////
/// Close the GUI frame

void TGLSAFrame::CloseWindow()
{
   // Ask our owning viewer to close
   // Has to be deferred so that our GUI event thread can process this event
   // and emit signals - otherwise deleted object is called to emit events
   // Not very nice but seems to be only reliable way to close down
   TTimer::SingleShot(50, "TGLSAViewer", &fViewer, "Close()");
}
