// @(#)root/gl:$Id$
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

#include "TGLLogicalShape.h"

class TBuffer3DSphere;

class TGLSphere : public TGLLogicalShape
{
private:
   Double_t fRadius; // Sphere radius

public:
   TGLSphere(const TBuffer3DSphere &buffer);

   virtual UInt_t DLOffset(Short_t lod) const;

   virtual ELODAxes SupportedLODAxes() const { return kLODAxesAll; }
   virtual Short_t  QuantizeShapeLOD(Short_t shapeLOD, Short_t combiLOD) const;
   virtual void     DirectDraw(TGLRnrCtx & rnrCtx) const;

   ClassDef(TGLSphere,0); // a spherical logical shape
};

#endif
