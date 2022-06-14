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

/** \class TGLAdapter
\ingroup opengl
Allow plot-painters to be used for gl-inpad and gl-viewer.
*/

ClassImp(TGLAdapter);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TGLAdapter::TGLAdapter(Int_t glDevice)
               : fGLDevice(glDevice)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set as current GL context.

Bool_t TGLAdapter::MakeCurrent()
{
   return fGLDevice != -1 && gGLManager->MakeCurrent(fGLDevice);
}

////////////////////////////////////////////////////////////////////////////////
/// Swap front/back buffers.

void TGLAdapter::SwapBuffers()
{
   if (fGLDevice != -1)
      gGLManager->Flush(fGLDevice);
}

////////////////////////////////////////////////////////////////////////////////
/// Mark gl-device for later copying into x-pixmap.

void TGLAdapter::MarkForDirectCopy(Bool_t isDirect)
{
   gGLManager->MarkForDirectCopy(fGLDevice, isDirect);
}

////////////////////////////////////////////////////////////////////////////////
/// Read gl buffer into x-pixmap.

void TGLAdapter::ReadGLBuffer()
{
   gGLManager->ReadGLBuffer(fGLDevice);
}

////////////////////////////////////////////////////////////////////////////////
/// Extract viewport from gl.

void TGLAdapter::ExtractViewport(Int_t *vp)const
{
   gGLManager->ExtractViewport(fGLDevice, vp);
}

////////////////////////////////////////////////////////////////////////////////
/// Select off-screen device for rendering.

void TGLAdapter::SelectOffScreenDevice()
{
   gGLManager->SelectOffScreenDevice(fGLDevice);
}
