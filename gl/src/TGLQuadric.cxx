// @(#)root/gl:$Name:  $:$Id: TGLQuadric.cxx

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLQuadric.h"
#include "TGLIncludes.h"
#include "TError.h"

ClassImp(TGLQuadric)

//______________________________________________________________________________
TGLQuadric::TGLQuadric() 
{
   fgQuad = 0;
}

//______________________________________________________________________________
TGLQuadric::~TGLQuadric() 
{ 
   if (fgQuad) {
      gluDeleteQuadric(fgQuad); 
   }
}

//______________________________________________________________________________
GLUquadric * TGLQuadric::Get()
{
   if (!fgQuad) {
      fgQuad = gluNewQuadric();
      if (!fgQuad) {
         Error("TGLQuadric::Get", "create failed");
      } else {
         gluQuadricOrientation(fgQuad, (GLenum)GLU_OUTSIDE);
         gluQuadricNormals(fgQuad, (GLenum)GLU_SMOOTH);
      }
   }
   return fgQuad;
}
