// @(#)root/gl:$Name:  $:$Id: TGLPolyLine.h,v 1.1.1.1 2007/04/04 16:01:43 mtadel Exp $
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

#ifndef ROOT_TGLPolyLine
#define ROOT_TGLPolyLine

#ifndef ROOT_TGLLogicalShape
#include "TGLLogicalShape.h"
#endif

#include <vector>

class TBuffer3D;

class TGLPolyLine : public TGLLogicalShape
{
private:
   std::vector<Double_t> fVertices;
   Double_t              fLineWidth;
protected:
   void DirectDraw(TGLRnrCtx & rnrCtx) const;

public:
   TGLPolyLine(const TBuffer3D & buffer);

   ClassDef(TGLPolyLine,0) // a polyline logical shape
};

#endif
