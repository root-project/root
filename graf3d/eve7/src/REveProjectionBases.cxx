// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveProjectionBases.hxx>
#include <ROOT/REveProjectionManager.hxx>
#include <ROOT/REveManager.hxx>

#include <cassert>

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class REveProjectable
\ingroup REve
Abstract base-class for non-linear projectable objects.

Via ProjectedClass(const REveProjection* p) method it returns a
TClass instance for the projected class and keeps references to the
projected objects.

It is assumed that all classes deriving from REveProjectable are also
derived from REveElement.

See also REveProjectionManager::ImportElements().
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveProjectable::REveProjectable()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor. Does shallow copy

REveProjectable::REveProjectable(const REveProjectable &)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.
/// Force projected replicas to unreference *this, then destroy them.

REveProjectable::~REveProjectable()
{
   // FIXME: fomr SL: how list becomes empty here???

   while ( ! fProjectedList.empty())
   {
      REveProjected* p = fProjectedList.front();

      p->UnRefProjectable(this);
      REveElement* el = p->GetProjectedAsElement();
      assert(el);
      {
         // FIXME: SL: no any globals !!!
         REX::gEve->PreDeleteElement(el);
         delete el;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Optimized destroy of projected elements with condition
/// there is only one parent for projected element. Method is
/// called from REveElement::Annihilate().

void REveProjectable::AnnihilateProjecteds()
{
   for (auto &&proj : fProjectedList) {
      proj->UnRefProjectable(this, kFALSE);
      proj->GetProjectedAsElement()->Annihilate();
   }
   fProjectedList.clear();
}

////////////////////////////////////////////////////////////////////////////////

void REveProjectable::ClearProjectedList()
{
   fProjectedList.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Add the projected elements to the set, dyn-casting them to
/// REveElement.

void REveProjectable::AddProjectedsToSet(std::set<REveElement*>& set)
{
   for (auto &&proj : fProjectedList)
      set.insert(proj->GetProjectedAsElement());
}

////////////////////////////////////////////////////////////////////////////////
/// Set visualization parameters of projecteds.
/// Use element el as model. If el == 0 (default), this casted to
/// REveElement is used.

void REveProjectable::PropagateVizParams(REveElement* el)
{
   if (el == nullptr)
      el = dynamic_cast<REveElement*>(this);

   for (auto &&proj : fProjectedList)
      proj->GetProjectedAsElement()->CopyVizParams(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Set render state of projecteds.

void REveProjectable::PropagateRenderState(Bool_t rnr_self, Bool_t rnr_children)
{
   for (auto &&proj : fProjectedList) {
      if (proj->GetProjectedAsElement()->SetRnrSelfChildren(rnr_self, rnr_children))
         proj->GetProjectedAsElement()->ElementChanged();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set main color of projecteds if their color is the same as old_color.

void REveProjectable::PropagateMainColor(Color_t color, Color_t old_color)
{
   for (auto &&proj : fProjectedList) {
      if (proj->GetProjectedAsElement()->GetMainColor() == old_color)
         proj->GetProjectedAsElement()->SetMainColor(color);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set main transparency of projecteds if their transparency is the
/// same as the old one.

void REveProjectable::PropagateMainTransparency(Char_t t, Char_t old_t)
{
   for (auto &&proj : fProjectedList) {
      if (proj->GetProjectedAsElement()->GetMainTransparency() == old_t)
         proj->GetProjectedAsElement()->SetMainTransparency(t);
   }
}

/** \class REveProjected
\ingroup REve
Abstract base class for classes that hold results of a non-linear
projection transformation.

It is assumed that all classes deriving from REveProjected are also
derived from REveElement.
*/

////////////////////////////////////////////////////////////////////////////////
/// Destructor.
/// If fProjectable is non-null, *this is removed from its list of
/// projected replicas.

REveProjected::~REveProjected()
{
   if (fProjectable) fProjectable->RemoveProjected(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns this projected dynamic-casted to REveElement.
/// This is needed as class REveProjected is used as secondary
/// inheritance.

REveElement* REveProjected::GetProjectedAsElement()
{
   return dynamic_cast<REveElement*>(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Sets projection manager and reference in the projectable object. Method called
/// immediately after default constructor.
/// See also REveProjectionManager::ImportElements().

void REveProjected::SetProjection(REveProjectionManager* mng, REveProjectable* model)
{
   fManager   = mng;
   if (fProjectable) fProjectable->RemoveProjected(this);
   fProjectable = model;
   if (fProjectable) fProjectable->AddProjected(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove reference to projectable.

void REveProjected::UnRefProjectable(REveProjectable* assumed_parent, bool notifyParent)
{
   static const REveException eH("REveProjected::UnRefProjectable ");

   R__ASSERT(fProjectable == assumed_parent);

   if (notifyParent) fProjectable->RemoveProjected(this);
   fProjectable = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Set depth coordinate for the element.
/// Bounding-box should also be updated.
/// If projection type is 3D, this only sets fDepth member.

void REveProjected::SetDepth(Float_t d)
{
   if (fManager->GetProjection()->Is2D()) {
      SetDepthLocal(d);
   } else {
      fDepth = d;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Utility function to update the z-values of the bounding-box.
/// As this is an abstract interface, the element and bbox pointers
/// must be passed from outside.

void REveProjected::SetDepthCommon(Float_t d, REveElement* el, Float_t* bbox)
{
   Float_t delta = d - fDepth;
   fDepth = d;
   if (bbox) {
      bbox[4] += delta;
      bbox[5] += delta;
      el->StampTransBBox();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Base-class implementation -- just sets fDepth.

void REveProjected::SetDepthLocal(Float_t d)
{
   fDepth = d;
}
