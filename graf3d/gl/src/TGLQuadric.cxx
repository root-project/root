// @(#)root/gl:$Id$

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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLOutput                                                            //
//                                                                      //
// Wrapper class for GLU quadric shape drawing object. Lazy creation of //
// internal GLU raw quadric on first call to TGLQuadric::Get()          //
//////////////////////////////////////////////////////////////////////////

ClassImp(TGLQuadric)

//______________________________________________________________________________
TGLQuadric::TGLQuadric() :
   fQuad(0)
{
   // Construct quadric
}

//______________________________________________________________________________
TGLQuadric::~TGLQuadric()
{
   // Destroy quadric
   if (fQuad) {
      gluDeleteQuadric(fQuad);
   }
}

//______________________________________________________________________________
GLUquadric * TGLQuadric::Get()
{
   // Get the internal raw GLU qaudric object. Created on first call.
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
