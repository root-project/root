// @(#)root/gl:$Name:  $:$Id: TGLSphere.h,v 1.1 2006/02/20 11:10:06 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004
// NOTE: This code moved from obsoleted TGLSceneObject.h / .cxx - see these
// attic files for previous CVS history

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLSphere
#define ROOT_TGLSphere

#ifndef ROOT_TGLLogicalShape
#include "TGLLogicalShape.h"
#endif

class TBuffer3DSphere;

class TGLSphere : public TGLLogicalShape 
{
private:
   Double_t fRadius; // Sphere radius

protected:
   void DirectDraw(const TGLDrawFlags & flags) const;  

public:
   TGLSphere(const TBuffer3DSphere &buffer);

   virtual ELODAxes SupportedLODAxes() const { return kLODAxesAll; }
   ClassDef(TGLSphere,0) // a spherical logical shape
};

#endif
