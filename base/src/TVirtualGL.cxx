// @(#)root/base:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   05/03/97

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-* TVirtualGL class *-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                     ================
//*-*
//*-*   TGLKernel class defines the interface for OpenGL commands and utilities
//*-*   Those are defined with GL/gl and GL/glu include directories
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

#include "TVirtualGL.h"

TVirtualGL *gVirtualGL=0;

//____________________________________________________________________________
TVirtualGL::TVirtualGL()
{
   fColorIndx     = 0;
   fRootLight     = kFALSE;
   fTrueColorMode = kFALSE;
   fFaceFlag      = kCCW;
}
