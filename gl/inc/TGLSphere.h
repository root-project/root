// @(#)root/gl:$Name:  $:$Id: TGLSphere.h,v 1.2 2007/02/04 17:39:44 brun Exp $
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
   void DirectDraw(TGLRnrCtx & rnrCtx) const;

public:
   TGLSphere(const TBuffer3DSphere &buffer);

   virtual Int_t  DLCacheSize()         const  { return 14; }
   virtual UInt_t DLOffset(Short_t lod) const;

   virtual ELODAxes SupportedLODAxes() const { return kLODAxesAll; }
   virtual Short_t  QuantizeShapeLOD(Short_t shapeLOD, Short_t combiLOD) const;

   ClassDef(TGLSphere,0) // a spherical logical shape
};

#endif
