// @(#)root/base:$Id$
// Author: Fons Rademakers   08/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAtt3D
#define ROOT_TAtt3D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAtt3D                                                               //
//                                                                      //
// Use this attribute class when an object should have 3D capabilities. //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif


class TAtt3D {

public:
   TAtt3D() { }
   virtual ~TAtt3D() { }

   virtual void Sizeof3D() const;

   ClassDef(TAtt3D,1)  //3D attributes
};

#endif

