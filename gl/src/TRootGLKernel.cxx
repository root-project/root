// @(#)root/gl:$Name:  $:$Id: TRootGLKernel.cxx,v 1.1.1.1 2000/05/16 17:00:47 rdm Exp $
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
#ifndef R__OPENINVENTOR
#include "TRootGLViewer.h"
#else
#include "TRootOIViewer.h"
#endif


// Force creation of TRootGLKernel when shared library will be loaded.
static TRootGLKernel gGLKernelCreator;


//______________________________________________________________________________
TGLViewerImp *TRootGLKernel::CreateGLViewerImp(TPadOpenGLView *p, const char *title,
                                               UInt_t width, UInt_t height)
{
#ifndef R__OPENINVENTOR
   return new TRootGLViewer(p, title, width, height);
#else
   return new TRootOIViewer(p, title, width, height);
#endif
}
