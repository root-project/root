// @(#)root/gl:$Name$:$Id$
// Author: Fons Rademakers   04/03/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootGLKernel                                                        //
//                                                                      //
// The TRootGLKernel class overrides the CreateGLViewerImp method to    //
// return a ROOT native GUI version of the GLViewer.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TRootGLKernel.h"
#include "TRootGLViewer.h"


// Force creation of TRootGLKernel when shared library will be loaded.
static TRootGLKernel gGLKernelCreator;


//______________________________________________________________________________
TGLViewerImp *TRootGLKernel::CreateGLViewerImp(TPadOpenGLView *p, const char *title,
                                               UInt_t width, UInt_t height)
{
   return new TRootGLViewer(p, title, width, height);
}
