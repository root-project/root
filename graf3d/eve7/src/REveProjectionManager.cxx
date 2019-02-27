// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveProjectionManager.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveProjectionBases.hxx>
#include <ROOT/REveCompound.hxx>

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TClass.h"

#include <list>

/** \class REveProjectionManager
\ingroup REve
Manager class for steering of projections and managing projected objects.

Recursively projects REveElement's and draws axis in the projected
scene. It enables to interactively set REveProjection parameters
and updates projected scene accordingly.
*/

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveProjectionManager::REveProjectionManager(REveProjection::EPType_e type):
   REveElement("REveProjectionManager","")
{
   for (Int_t i = 0; i < REveProjection::kPT_End; ++i)
      fProjections[i] = nullptr;

   if (type != REveProjection::kPT_Unknown)
      SetProjection(type);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.
/// Destroys also dependent elements.

REveProjectionManager::~REveProjectionManager()
{
   for (Int_t i = 0; i < REveProjection::kPT_End; ++i) {
      delete fProjections[i];
   }
   while ( ! fDependentEls.empty()) {
      fDependentEls.front()->Destroy();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add el as dependent element.

void REveProjectionManager::AddDependent(REveElement *el)
{
   fDependentEls.push_back(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove el as dependent element.

void REveProjectionManager::RemoveDependent(REveElement *el)
{
   fDependentEls.remove(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Updates name to have consistent information with projection.

void REveProjectionManager::UpdateName()
{
   if (fProjection->Is2D())
      SetName(Form ("%s (%3.1f)", fProjection->GetName(), fProjection->GetDistortion()*1000));
   else
      SetName(fProjection->GetName());
}

////////////////////////////////////////////////////////////////////////////////
/// Set projection type and distortion.

void REveProjectionManager::SetProjection(REveProjection::EPType_e type)
{
   static const REveException eH("REveProjectionManager::SetProjection ");

   if (fProjections[type] == 0)
   {
      switch (type)
      {
         case REveProjection::kPT_RPhi:
         {
            fProjections[type] = new REveRPhiProjection();
            break;
         }
         case REveProjection::kPT_RhoZ:
         {
            fProjections[type] = new REveRhoZProjection();
            break;
         }
         case REveProjection::kPT_3D:
         {
            fProjections[type] = new REve3DProjection();
            break;
         }
         default:
            throw eH + "projection type not valid.";
            break;
      }
   }

   if (fProjection && fProjection->Is2D() != fProjections[type]->Is2D())
   {
      throw eH + "switching between 2D and 3D projections not implemented.";
   }

   fProjection = fProjections[type];
   fProjection->SetCenter(fCenter);
   UpdateName();
}

////////////////////////////////////////////////////////////////////////////////
/// Set projection center and rebuild projected scene.

void REveProjectionManager::SetCenter(Float_t x, Float_t y, Float_t z)
{
   fCenter.Set(x, y, z);
   fProjection->SetCenter(fCenter);
   ProjectChildren();
}

////////////////////////////////////////////////////////////////////////////////
/// React to element being pasted or dnd-ed.
/// Return true if redraw is needed (virtual method).

Bool_t REveProjectionManager::HandleElementPaste(REveElement* el)
{
   List_t::size_type n_children  = fChildren.size();
   ImportElements(el);
   return n_children != fChildren.size();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if element el should be imported.
///
/// Behaviour depends on the value of the fImportEmpty member:
///   false - el or any of its children must be projectable (default);
///   true  - always import.

Bool_t REveProjectionManager::ShouldImport(REveElement *el)
{
   if (fImportEmpty)
      return kTRUE;

   if (el->IsA()->InheritsFrom(TClass::GetClass<REveProjectable>()))
      return kTRUE;
   for (auto &c: el->RefChildren())
      if (ShouldImport(c))
         return kTRUE;
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Update dependent elements' bounding box and mark scenes
/// containing element root or its children as requiring a repaint.

void REveProjectionManager::UpdateDependentElsAndScenes(REveElement* root)
{
   for (List_i i=fDependentEls.begin(); i!=fDependentEls.end(); ++i)
   {
      TAttBBox* bbox = dynamic_cast<TAttBBox*>(*i);
      if (bbox)
         bbox->ComputeBBox();
   }

   List_t scenes;
   root->CollectScenes(scenes);
   if (root == this)
   {
      for (auto &n : fNieces) n->CollectScenes(scenes);
   }

   REX::gEve->ScenesChanged(scenes);
}

////////////////////////////////////////////////////////////////////////////////
/// If el is REveProjectable add projected instance else add plain
/// REveElementList to parent. Call the same function on el's
/// children.
///
/// Returns the projected replica of el. Can be 0, if el and none of
/// its children are projectable.

REveElement* REveProjectionManager::ImportElementsRecurse(REveElement* el,
                                                          REveElement* parent)
{
   static const REveException eh("REveProjectionManager::ImportElementsRecurse ");

   REveElement *new_el = nullptr;

   if (ShouldImport(el))
   {
      REveProjected   *new_pr = nullptr;
      REveProjectable *pble   = dynamic_cast<REveProjectable*>(el);
      if (pble)
      {
         new_el = (REveElement*) pble->ProjectedClass(fProjection)->New();
         new_pr = dynamic_cast<REveProjected*>(new_el);
         new_pr->SetProjection(this, pble);
         new_pr->SetDepth(fCurrentDepth);
      }
      else
      {
         new_el = new REveElement;
      }
      new_el->SetName (Form("%s [P]", el->GetCName()));
      new_el->SetTitle(Form("Projected replica.\n%s", el->GetCTitle()));
      new_el->SetRnrSelf     (el->GetRnrSelf());
      new_el->SetRnrChildren (el->GetRnrChildren());
      new_el->SetPickable    (el->IsPickable());

      parent->AddElement(new_el);

      REveCompound *cmpnd    = dynamic_cast<REveCompound*>(el);
      REveCompound *cmpnd_pr = dynamic_cast<REveCompound*>(new_el);
      for (auto &c: el->RefChildren()) {
         REveElement *child_pr = ImportElementsRecurse(c, new_el);
         if (cmpnd && c->GetCompound() == cmpnd)
            child_pr->SetCompound(cmpnd_pr);
      }
   }

   return new_el;
}

////////////////////////////////////////////////////////////////////////////////
/// Recursively import elements and apply projection to the newly
/// imported objects.
///
/// If ext_list is not 0 the new element is also added to the list.
/// This simplifies construction of complex views where projected
/// elements are distributed into several scenes for optimization of
/// updates and rendering.
///
/// Returns the projected replica of el. Can be 0, if el and none of
/// its children are projectable.

REveElement* REveProjectionManager::ImportElements(REveElement* el,
                                                   REveElement* ext_list)
{
   REveElement* new_el = ImportElementsRecurse(el, ext_list ? ext_list : this);
   if (new_el)
   {
      AssertBBox();
      ProjectChildrenRecurse(new_el);
      AssertBBoxExtents(0.1);
      StampTransBBox();

      UpdateDependentElsAndScenes(new_el);

      if (ext_list)
         AddNiece(new_el);
   }
   return new_el;
}

////////////////////////////////////////////////////////////////////////////////
/// Recursively import elements and apply projection to the newly
/// imported objects.
///
/// The proj_parent argument should be a projected replica of parent
/// of element 'el'. This allows to insert projected children of
/// a given element when they are added after the projection has
/// been already performed on the parent.
/// This is called from REveElement::ProjectChild().
///
/// Returns the projected replica of el. Can be 0, if el and none of
/// its children are projectable.

REveElement* REveProjectionManager::SubImportElements(REveElement* el,
                                                      REveElement* proj_parent)
{
   REveElement* new_el = ImportElementsRecurse(el, proj_parent);
   if (new_el)
   {
      AssertBBox();
      ProjectChildrenRecurse(new_el);
      AssertBBoxExtents(0.1);
      StampTransBBox();

      UpdateDependentElsAndScenes(new_el);
   }
   return new_el;
}

////////////////////////////////////////////////////////////////////////////////
/// Recursively import children elements of el and apply projection
/// to the newly imported objects.
///
/// The proj_parent argument should be a projected replica of
/// element 'el'. This allows to insert projected children of
/// a given element when they are added after the projection has
/// been already performed on the parent.
/// This is called from REveElement::ProjectChild().
///
/// Returns the projected replica of el. Can be 0, if el and none of
/// its children are projectable.

Int_t REveProjectionManager::SubImportChildren(REveElement* el, REveElement* proj_parent)
{
   List_t new_els;
   for (auto &c: el->RefChildren()) {
      auto new_el = ImportElementsRecurse(c, proj_parent);
      if (new_el)
         new_els.push_back(new_el);
   }

   if (!new_els.empty())
   {
      AssertBBox();
      for (auto &nel: new_els)
         ProjectChildrenRecurse(nel);
      AssertBBoxExtents(0.1);
      StampTransBBox();

      UpdateDependentElsAndScenes(proj_parent);
   }
   return (Int_t) new_els.size();
}

////////////////////////////////////////////////////////////////////////////////
/// Project el (via REveProjected::UpdateProjection()) and recurse
/// through el's children.
/// Bounding-box is updated along the recursion.

void REveProjectionManager::ProjectChildrenRecurse(REveElement* el)
{
   REveProjected* pted = dynamic_cast<REveProjected*>(el);
   if (pted)
   {
      pted->UpdateProjection();
      TAttBBox* bb = dynamic_cast<TAttBBox*>(pted);
      if (bb)
      {
         Float_t* b = bb->AssertBBox();
         BBoxCheckPoint(b[0], b[2], b[4]);
         BBoxCheckPoint(b[1], b[3], b[5]);
      }
      el->ElementChanged(kFALSE);
   }

   for (auto &c : el->RefChildren())  ProjectChildrenRecurse(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Project all children recursively, update bounding-box and notify
/// EveManger about the scenes that have been changed.

void REveProjectionManager::ProjectChildren()
{
   BBoxInit();

   for (auto &c : fChildren)  ProjectChildrenRecurse(c);

   for (auto &n : fNieces)    ProjectChildrenRecurse(n);

   AssertBBoxExtents(0.1);
   StampTransBBox();

   UpdateDependentElsAndScenes(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TAttBBox; fill bounding-box information.
///
/// The bounding-box information is kept coherent during addition of
/// projected elements and projection parameter updates. This is
/// called only in case the manager has not been populated at all.

void REveProjectionManager::ComputeBBox()
{
   static const REveException eH("REveProjectionManager::ComputeBBox ");

   if ( ! HasChildren() && ! HasNieces())
   {
      BBoxZero();
      return;
   }

   BBoxInit();
}
