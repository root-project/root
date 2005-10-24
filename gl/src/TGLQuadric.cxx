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
TGLQuadric::TGLQuadric() :
   fQuad(0)
{
}

//______________________________________________________________________________
TGLQuadric::~TGLQuadric() 
{ 
   if (fQuad) {
      gluDeleteQuadric(fQuad); 
   }
}

//______________________________________________________________________________
GLUquadric * TGLQuadric::Get()
{
   if (!fQuad) {
      fQuad = gluNewQuadric();
      if (!fQuad) {
         Error("TGLQuadric::Get", "create failed");
      } else {
         gluQuadricOrientation(fQuad, (GLenum)GLU_OUTSIDE);
         gluQuadricNormals(fQuad, (GLenum)GLU_SMOOTH);
      }
   }
   return fQuad;
}
