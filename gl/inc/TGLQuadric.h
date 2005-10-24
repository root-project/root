// @(#)root/gl:$Name:  $:$Id: TGLQuadric.h
// Author:  Richard Maunder  16/09/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLQuadric
#define ROOT_TGLQuadric

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

/*************************************************************************
 * TGLQuadric - Wrapper for C GLUquadric to provide delayed creation and
 * autodeletion - can be used as static member of other objects
 *
 *************************************************************************/

class GLUquadric;

class TGLQuadric
{
private:
   GLUquadric * fQuad;
public:
   TGLQuadric();
   virtual ~TGLQuadric(); // ClassDef introduces virtuals
   
   GLUquadric * Get();

   ClassDef(TGLQuadric,0) // GL quadric object
};

#endif

