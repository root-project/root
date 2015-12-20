// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLPShapeRef.h"
#include "TGLPhysicalShape.h"

/** \class TGLPShapeRef
\ingroup opengl
Base class for references to TGLPysicalShape that need to be notified
when the shape is destroyed.
Could also deliver 'change' notifications.
*/

ClassImp(TGLPShapeRef);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TGLPShapeRef::TGLPShapeRef() :
   fNextPSRef (0),
   fPShape    (0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with known shape - reference it.

TGLPShapeRef::TGLPShapeRef(TGLPhysicalShape * shape) :
   fNextPSRef (0),
   fPShape    (0)
{
   SetPShape(shape);
}
////////////////////////////////////////////////////////////////////////////////
/// Destructor - unreference the shape if set.

TGLPShapeRef::~TGLPShapeRef()
{
   SetPShape(0);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the shape. Unreference the old and reference the new.
/// This is virtual so that sub-classes can perform other tasks
/// on change. This function should be called first from there.
///
/// This is also called from destructor of the referenced physical
/// shape with 0 argument.

void TGLPShapeRef::SetPShape(TGLPhysicalShape * shape)
{
   if (fPShape)
      fPShape->RemoveReference(this);
   fPShape = shape;
   if (fPShape)
      fPShape->AddReference(this);
}

////////////////////////////////////////////////////////////////////////////////
/// This is called from physical shape when it is modified.
/// Sub-classes can override and take appropriate action.

void TGLPShapeRef::PShapeModified()
{
}
