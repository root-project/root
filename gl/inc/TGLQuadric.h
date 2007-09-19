// @(#)root/gl:$Id$
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLOutput                                                            //
//                                                                      //
// Wrapper class for GLU quadric shape drawing object. Lazy creation of //
// internal GLU raw quadric on first call to TGLQuadric::Get()          //
//////////////////////////////////////////////////////////////////////////

class GLUquadric;

class TGLQuadric
{
private:
   GLUquadric * fQuad;

protected:
   TGLQuadric(const TGLQuadric& glq) : fQuad(glq.fQuad) { }
   TGLQuadric& operator=(const TGLQuadric& glq)
     { if(this!=&glq) fQuad=glq.fQuad; return *this; }

public:
   TGLQuadric();
   virtual ~TGLQuadric(); // ClassDef introduces virtuals

   GLUquadric * Get();

   ClassDef(TGLQuadric,0) // GL quadric object
};

#endif

