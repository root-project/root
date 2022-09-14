// @(#)root/gl:$Id$
// Author:  Alja Mrak-Tadel  06/2006

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPShapeObj
#define ROOT_TGLPShapeObj

#include <TObject.h>

class TGLPhysicalShape;
class TGLViewer;

class TGLPShapeObj : public TObject
{
public:
   TGLPhysicalShape *fPShape;
   TGLViewer        *fViewer;

   TGLPShapeObj() : TObject(), fPShape(nullptr), fViewer(nullptr) {}
   TGLPShapeObj(TGLPhysicalShape* sh, TGLViewer* v) :
      TObject(), fPShape(sh), fViewer(v) {}
   virtual ~TGLPShapeObj() {}

   virtual const char* GetName() const { return "Selected"; }

private:
   TGLPShapeObj(const TGLPShapeObj &); // Not implemented
   TGLPShapeObj& operator=(const TGLPShapeObj &); // Not implemented

   ClassDef(TGLPShapeObj, 0) // This object wraps TGLPhysicalShape (not a TObject)
};

#endif
