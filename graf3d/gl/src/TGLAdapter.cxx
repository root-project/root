// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLAdapter.h"

//______________________________________________________________________________
//
// Allow plot-painters to be used for gl-inpad and gl-viewer.

ClassImp(TGLAdapter)

//______________________________________________________________________________
TGLAdapter::TGLAdapter(Int_t glDevice)
               : fGLDevice(glDevice)
{
   // Constructor.
}

//______________________________________________________________________________
Bool_t TGLAdapter::MakeCurrent()
{
   // Set as current GL contet.
   return fGLDevice != -1 && gGLManager->MakeCurrent(fGLDevice);
}

//______________________________________________________________________________
void TGLAdapter::SwapBuffers()
{
   // Swap front/back buffers.
   if (fGLDevice != -1)
      gGLManager->Flush(fGLDevice);
}

//______________________________________________________________________________
void TGLAdapter::MarkForDirectCopy(Bool_t isDirect)
{
   // Mark gl-device for later copying into x-pixmap.
   gGLManager->MarkForDirectCopy(fGLDevice, isDirect);
}

//______________________________________________________________________________
void TGLAdapter::ReadGLBuffer()
{
   // Read gl buffer into x-pixmap.
   gGLManager->ReadGLBuffer(fGLDevice);
}

//______________________________________________________________________________
void TGLAdapter::ExtractViewport(Int_t *vp)const
{
   // Extract viewport from gl.
   gGLManager->ExtractViewport(fGLDevice, vp);
}

//______________________________________________________________________________
void TGLAdapter::SelectOffScreenDevice()
{
   // Select off-screen device for rendering.
   gGLManager->SelectOffScreenDevice(fGLDevice);
}
