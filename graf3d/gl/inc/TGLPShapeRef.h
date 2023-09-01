// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPShapeRef
#define ROOT_TGLPShapeRef

#include <Rtypes.h>

class TGLPhysicalShape;

class TGLPShapeRef
{
   friend class TGLPhysicalShape;
private:
   TGLPShapeRef(const TGLPShapeRef&);            // Not implemented
   TGLPShapeRef& operator=(const TGLPShapeRef&); // Not implemented

   TGLPShapeRef * fNextPSRef;  // Internal pointer to the next reference (used by TGLPhysicalShape directly).

protected:
   TGLPhysicalShape * fPShape; // Pointer to referenced physical shape.

public:
   TGLPShapeRef();
   TGLPShapeRef(TGLPhysicalShape * shape);
   virtual ~TGLPShapeRef();

   TGLPhysicalShape * GetPShape() const { return fPShape; }
   virtual void SetPShape(TGLPhysicalShape * shape);
   virtual void PShapeModified();

   ClassDef(TGLPShapeRef, 0); // Reference to a TGLPhysicalShape object.
}; // endclass TGLPShapeRef


#endif
