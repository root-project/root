// @(#)root/gl:$Name$:$Id$
// Author: Fons Rademakers   04/03/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootGLKernel
#define ROOT_TRootGLKernel


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootGLKernel                                                        //
//                                                                      //
// The TRootGLKernel class overrides the CreateGLViewerImp method to    //
// return a ROOT native GUI version of the GLViewer.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGLKernel
#include "TGLKernel.h"
#endif

class TPadOpenGLView;


class TRootGLKernel : public TGLKernel {

public:
   virtual TGLViewerImp *CreateGLViewerImp(TPadOpenGLView *p, const char *title,
                                           UInt_t width, UInt_t height);
};

#endif
