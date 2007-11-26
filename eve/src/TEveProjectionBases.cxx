// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEvePolygonSetProjected.h>
#include <TEveProjectionBases.h>

//______________________________________________________________________________
// TEveProjectable
//
// Abstract base-class for non-linear projectable objects.
//
// Via ProjectedClass() method it returns a TClass instance for the
// projected class and keeps references to the projected objects.
//
// See also TEveProjectionManager::ImportElements().

ClassImp(TEveProjectable)

//______________________________________________________________________________
TEveProjectable::TEveProjectable()
{
   // Comstructor.
}

//______________________________________________________________________________
TEveProjectable::~TEveProjectable()
{
   // Destructor.
   // Force projected replicas to unreference *this.

   while ( ! fProjectedList.empty())
   {
      fProjectedList.front()->UnRefProjectable(this);
   }
}


//______________________________________________________________________________
// TEveProjected
//
// Abstract base class for classes that hold results of a non-linear
// projection transformation.
//

ClassImp(TEveProjected)

//______________________________________________________________________________
TEveProjected::TEveProjected() :
   fProjector   (0),
   fProjectable (0),
   fDepth       (0)
{
   // Constructor.
}

//______________________________________________________________________________
TEveProjected::~TEveProjected()
{
   // Destructor.
   // If fProjectable is non-null, *this is removed from its list of
   // projected replicas.

   if (fProjectable) fProjectable->RemoveProjected(this);
}

//______________________________________________________________________________
void TEveProjected::SetProjection(TEveProjectionManager* proj, TEveProjectable* model)
{
   fProjector   = proj;
   if (fProjectable) fProjectable->RemoveProjected(this);
   fProjectable = model;
   if (fProjectable) fProjectable->AddProjected(this);
}

//______________________________________________________________________________
void TEveProjected::UnRefProjectable(TEveProjectable* assumed_parent)
{
   static const TEveException eH("TEveProjected::UnRefProjectable ");

   if (fProjectable != assumed_parent) {
      Warning(eH, "mismatch between assumed and real model. This is a bug.");
      assumed_parent->RemoveProjected(this);
      return;
   }

   if (fProjectable) {
      fProjectable->RemoveProjected(this);
      fProjectable = 0;
   }
}
