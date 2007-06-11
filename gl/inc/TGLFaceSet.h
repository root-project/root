// @(#)root/gl:$Name:  $:$Id: TGLFaceSet.h,v 1.1.1.1 2007/04/04 16:01:43 mtadel Exp $
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

#ifndef ROOT_TGLFaceSet
#define ROOT_TGLFaceSet

#ifndef ROOT_TGLLogicalShape
#include "TGLLogicalShape.h"
#endif
#ifndef ROOT_CsgOps
#include "CsgOps.h"
#endif

///////////////////////////////////////////////////////////////////////
class TGLFaceSet : public TGLLogicalShape
{
private:
   std::vector<Double_t> fVertices;
   std::vector<Double_t> fNormals;
   std::vector<Int_t>    fPolyDesc;
   UInt_t                fNbPols;

protected:
   void DirectDraw(TGLRnrCtx & rnrCtx) const;

public:
   TGLFaceSet(const TBuffer3D & buffer);

   void SetFromMesh(const RootCsg::TBaseMesh *m);

private:
   void GLDrawPolys()const;
   Int_t CheckPoints(const Int_t *source, Int_t *dest)const;
   static Bool_t Eq(const Double_t *p1, const Double_t *p2);
   void CalculateNormals();

   ClassDef(TGLFaceSet,0) // a faceset logical shape
};

#endif
