// Author: Richard Maunder   04/08/05

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TX3DFrame                                                            //
//                                                                      //
// Main frame for TViewerX3D                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TX3DFrame.h"
#include "TViewerX3D.h"

////////////////////////////////////////////////////////////////////////////////
/// TX3DFrame constructor

TX3DFrame::TX3DFrame(TViewerX3D & viewer, const TGWindow * win, UInt_t width, UInt_t height) :
   TGMainFrame(win, width, height),
   fViewer(viewer)
{
}

////////////////////////////////////////////////////////////////////////////////
/// TX3DFrame destructor

TX3DFrame::~TX3DFrame()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Process Message

Bool_t TX3DFrame::ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2)
{
   return fViewer.ProcessFrameMessage(msg, parm1, parm2);
}

////////////////////////////////////////////////////////////////////////////////
/// Close window

void TX3DFrame::CloseWindow()
{
   fViewer.Close();
}
