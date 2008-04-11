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

//______________________________________________________________________________
TX3DFrame::TX3DFrame(TViewerX3D & viewer, const TGWindow * win, UInt_t width, UInt_t height) :
   TGMainFrame(win, width, height),
   fViewer(viewer)
{
   // TX3DFrame constructor
}

//______________________________________________________________________________
TX3DFrame::~TX3DFrame()
{
   // TX3DFrame destructor
}

//______________________________________________________________________________
Bool_t TX3DFrame::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Process Message
   return fViewer.ProcessFrameMessage(msg, parm1, parm2);
}

//______________________________________________________________________________
void TX3DFrame::CloseWindow()
{
   // Close window
   fViewer.Close();
}
