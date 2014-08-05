// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveProjectionBases.h"
#include "TEveProjectionManager.h"
#include "TEveManager.h"

#include <cassert>

//==============================================================================
//==============================================================================
// TEveProjectable
//==============================================================================

//______________________________________________________________________________
//
// Abstract base-class for non-linear projectable objects.
//
// Via ProjectedClass(const TEveProjection* p) method it returns a
// TClass instance for the projected class and keeps references to the
// projected objects.
//
// It is assumed that all classes deriving from TEveProjectable are also
// derived from TEveElement.
//
// See also TEveProjectionManager::ImportElements().

ClassImp(TEveProjectable);

//______________________________________________________________________________
TEveProjectable::TEveProjectable()
{
   // Constructor.
}

//______________________________________________________________________________
TEveProjectable::~TEveProjectable()
{
   // Destructor.
   // Force projected replicas to unreference *this, then destroy them.

   while ( ! fProjectedList.empty())
   {
      TEveProjected* p = fProjectedList.front();
      p->UnRefProjectable(this);
      TEveElement* el = p->GetProjectedAsElement();
      assert(el);
      {
         gEve->PreDeleteElement(el);
         delete el;
      }
   }
}

//______________________________________________________________________________
void TEveProjectable::AnnihilateProjecteds()
{
   // Optimized destroy of projected elements with condition
   // there is only one parent for projected element. Method is
   // called from TEveElement::Annihilate().

   for (ProjList_i i=fProjectedList.begin(); i!=fProjectedList.end(); ++i)
   {
      (*i)->UnRefProjectable(this, kFALSE);
      (*i)->GetProjectedAsElement()->Annihilate();
   }
   fProjectedList.clear();
}

//______________________________________________________________________________
void TEveProjectable::ClearProjectedList()
{
   fProjectedList.clear();
}

//______________________________________________________________________________
void TEveProjectable::AddProjectedsToSet(std::set<TEveElement*>& set)
{
   // Add the projected elements to the set, dyn-casting them to
   // TEveElement.

   for (ProjList_i i=fProjectedList.begin(); i!=fProjectedList.end(); ++i)
   {
      set.insert((*i)->GetProjectedAsElement());
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveProjectable::PropagateVizParams(TEveElement* el)
{
   // Set visualization parameters of projecteds.
   // Use element el as model. If el == 0 (default), this casted to
   // TEveElement is used.

   if (el == 0)
      el = dynamic_cast<TEveElement*>(this);

   for (ProjList_i i=fProjectedList.begin(); i!=fProjectedList.end(); ++i)
   {
      (*i)->GetProjectedAsElement()->CopyVizParams(el);
   }
}

//______________________________________________________________________________
void TEveProjectable::PropagateRenderState(Bool_t rnr_self, Bool_t rnr_children)
{
   // Set render state of projecteds.

   for (ProjList_i i=fProjectedList.begin(); i!=fProjectedList.end(); ++i)
   {
      if ((*i)->GetProjectedAsElement()->SetRnrSelfChildren(rnr_self, rnr_children))
         (*i)->GetProjectedAsElement()->ElementChanged();
   }
}

//______________________________________________________________________________
void TEveProjectable::PropagateMainColor(Color_t color, Color_t old_color)
{
   // Set main color of projecteds if their color is the same as old_color.

   for (ProjList_i i=fProjectedList.begin(); i!=fProjectedList.end(); ++i)
   {
      if ((*i)->GetProjectedAsElement()->GetMainColor() == old_color)
         (*i)->GetProjectedAsElement()->SetMainColor(color);
   }
}

//______________________________________________________________________________
void TEveProjectable::PropagateMainTransparency(Char_t t, Char_t old_t)
{
   // Set main transparency of projecteds if their transparecy is the
   // same as the old one.

   for (ProjList_i i=fProjectedList.begin(); i!=fProjectedList.end(); ++i)
   {
      if ((*i)->GetProjectedAsElement()->GetMainTransparency() == old_t)
         (*i)->GetProjectedAsElement()->SetMainTransparency(t);
   }
}


//==============================================================================
//==============================================================================
// TEveProjected
//==============================================================================

//______________________________________________________________________________
//
// Abstract base class for classes that hold results of a non-linear
// projection transformation.
//
// It is assumed that all classes deriving from TEveProjected are also
// derived from TEveElement.

ClassImp(TEveProjected);

//______________________________________________________________________________
TEveProjected::TEveProjected() :
   fManager     (0),
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
TEveElement* TEveProjected::GetProjectedAsElement()
{
   // Returns this projected dynamic-casted to TEveElement.
   // This is needed as class TEveProjected is used as secondary
   // inheritance.

   return dynamic_cast<TEveElement*>(this);
}

//______________________________________________________________________________
void TEveProjected::SetProjection(TEveProjectionManager* mng, TEveProjectable* model)
{
   // Sets projection manager and reference in the projectable object. Method called
   // immediately after default constructor.
   // See also TEveProjectionManager::ImportElements().

   fManager   = mng;
   if (fProjectable) fProjectable->RemoveProjected(this);
   fProjectable = model;
   if (fProjectable) fProjectable->AddProjected(this);
}

//______________________________________________________________________________
void TEveProjected::UnRefProjectable(TEveProjectable* assumed_parent, bool notifyParent)
{
   // Remove reference to projectable.

   static const TEveException eH("TEveProjected::UnRefProjectable ");

   R__ASSERT(fProjectable == assumed_parent);

   if (notifyParent) fProjectable->RemoveProjected(this);
   fProjectable = 0;
}

//______________________________________________________________________________
void TEveProjected::SetDepth(Float_t d)
{
   // Set depth coordinate for the element.
   // Bounding-box should also be updated.
   // If projection type is 3D, this only sets fDepth member.

   if (fManager->GetProjection()->Is2D())
   {
      SetDepthLocal(d);
   }
   else
   {
      fDepth = d;
   }
}

//______________________________________________________________________________
void TEveProjected::SetDepthCommon(Float_t d, TEveElement* el, Float_t* bbox)
{
   // Utility function to update the z-values of the bounding-box.
   // As this is an abstract interface, the element and bbox pointers
   // must be passed from outside.

   Float_t delta = d - fDepth;
   fDepth = d;
   if (bbox) {
      bbox[4] += delta;
      bbox[5] += delta;
      el->StampTransBBox();
   }
}

//______________________________________________________________________________
void TEveProjected::SetDepthLocal(Float_t d)
{
   // Base-class implementation -- just sets fDepth.

   fDepth = d;
}
