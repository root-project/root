// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveElement.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveCompound.hxx>
#include <ROOT/REveTrans.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveSelection.hxx>
#include <ROOT/REveProjectionBases.hxx>
#include <ROOT/REveProjectionManager.hxx>
#include <ROOT/REveRenderData.hxx>

#include "TGeoMatrix.h"

#include "TClass.h"
#include "TPRegexp.h"
#include "TROOT.h"
#include "TColor.h"

#include "json.hpp"

#include <algorithm>

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class REveElement
\ingroup REve
Base class for REveUtil visualization elements, providing hierarchy
management, rendering control and list-tree item management.
*/

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

REveElement::REveElement() :
   fParents             (),
   fChildren            (),
   fCompound            (0),
   fVizModel            (0),
   fVizTag              (),
   fNumChildren         (0),
   fParentIgnoreCnt     (0),
   fDenyDestroy         (0),
   fDestroyOnZeroRefCnt (kTRUE),
   fRnrSelf             (kTRUE),
   fRnrChildren         (kTRUE),
   fCanEditMainColor    (kFALSE),
   fCanEditMainTransparency(kFALSE),
   fCanEditMainTrans    (kFALSE),
   fMainTransparency    (0),
   fMainColorPtr        (0),
   fMainTrans           (),
   fSource              (),
   fPickable            (kFALSE),
   fSelected            (kFALSE),
   fHighlighted         (kFALSE),
   fImpliedSelected     (0),
   fImpliedHighlighted  (0),
   fCSCBits             (0),
   fChangeBits          (0),
   fDestructing         (kNone)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveElement::REveElement(Color_t &main_color) :
   fParents             (),
   fChildren            (),
   fCompound            (nullptr),
   fVizModel            (nullptr),
   fVizTag              (),
   fNumChildren         (0),
   fParentIgnoreCnt     (0),
   fDenyDestroy         (0),
   fDestroyOnZeroRefCnt (kTRUE),
   fRnrSelf             (kTRUE),
   fRnrChildren         (kTRUE),
   fCanEditMainColor    (kFALSE),
   fCanEditMainTransparency(kFALSE),
   fCanEditMainTrans    (kFALSE),
   fMainTransparency    (0),
   fMainColorPtr        (&main_color),
   fMainTrans           (),
   fSource              (),
   fPickable            (kFALSE),
   fSelected            (kFALSE),
   fHighlighted         (kFALSE),
   fImpliedSelected     (0),
   fImpliedHighlighted  (0),
   fCSCBits             (0),
   fChangeBits          (0),
   fDestructing         (kNone)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor. Does shallow copy.
/// For deep-cloning and children-cloning, see:
/// ~~~ {.cpp}
///   REveElement* CloneElementRecurse(Int_t level)
///   void         CloneChildrenRecurse(REveElement* dest, Int_t level)
/// ~~~
/// 'TRef fSource' is copied but 'void* UserData' is NOT.
/// If the element is projectable, its projections are NOT copied.
///
/// Not implemented for most sub-classes, let us know.
/// Note that sub-classes of REveProjected are NOT and will NOT be copyable.

REveElement::REveElement(const REveElement& e) :
   fParents             (),
   fChildren            (),
   fCompound            (nullptr),
   fVizModel            (nullptr),
   fVizTag              (e.fVizTag),
   fNumChildren         (0),
   fParentIgnoreCnt     (0),
   fDenyDestroy         (0),
   fDestroyOnZeroRefCnt (e.fDestroyOnZeroRefCnt),
   fRnrSelf             (e.fRnrSelf),
   fRnrChildren         (e.fRnrChildren),
   fCanEditMainColor    (e.fCanEditMainColor),
   fCanEditMainTransparency(e.fCanEditMainTransparency),
   fCanEditMainTrans    (e.fCanEditMainTrans),
   fMainTransparency    (e.fMainTransparency),
   fMainColorPtr        (nullptr),
   fMainTrans           (),
   fSource              (e.fSource),
   fPickable            (e.fPickable),
   fSelected            (kFALSE),
   fHighlighted         (kFALSE),
   fImpliedSelected     (0),
   fImpliedHighlighted  (0),
   fCSCBits             (e.fCSCBits),
   fChangeBits          (0),
   fDestructing         (kNone)
{
   SetVizModel(e.fVizModel);
   // FIXME: from Sergey: one have to use other way to referencing main color
   if (e.fMainColorPtr)
      fMainColorPtr = (Color_t*)((const char*) this + ((const char*) e.fMainColorPtr - (const char*) &e));
   if (e.fMainTrans)
      fMainTrans = std::make_unique<REveTrans>(*e.fMainTrans.get());
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor. Do not call this method directly, either call Destroy() or
/// Annihilate(). See also DestroyElements() and AnnihilateElements() if you
/// need to delete all children of an element.

REveElement::~REveElement()
{
   if (fDestructing != kAnnihilate)
   {
      fDestructing = kStandard;
      RemoveElementsInternal();

      for (auto &p : fParents)
      {
         p->RemoveElementLocal(this);
         p->fChildren.remove(this);
         --(p->fNumChildren);
      }
   }

   fParents.clear();
}

ElementId_t REveElement::get_mother_id() const
{
   return fMother ? fMother->GetElementId() : 0;
}

ElementId_t REveElement::get_scene_id() const
{
   return fScene ? fScene->GetElementId() : 0;
}

void REveElement::assign_element_id_recurisvely()
{
   assert(fElementId == 0);

   REX::gEve->AssignElementId(this);
   for (auto &c : fChildren)
      c->assign_element_id_recurisvely();
}

void REveElement::assign_scene_recursively(REveScene* s)
{
   assert(fScene == 0);
   fScene = s;

   if (fDestructing == kNone && fScene && fScene->IsAcceptingChanges()) {
       s->SceneElementAdded(this);
   }
   for (auto &c : fChildren)
      c->assign_scene_recursively(s);
}

////////////////////////////////////////////////////////////////////////////////
/// Called before the element is deleted, thus offering the last chance
/// to detach from acquired resources and from the framework itself.
/// Here the request is just passed to REveManager.
/// If you override it, make sure to call base-class version.

void REveElement::PreDeleteElement()
{
   if (fElementId != 0) {
      REX::gEve->PreDeleteElement(this);
      fScene->SceneElementRemoved( fElementId);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Clone the element via copy constructor.
/// Should be implemented for all classes that require cloning support.

REveElement* REveElement::CloneElement() const
{
   return new REveElement(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Clone elements and recurse 'level' deep over children.
///  - If level ==  0, only the element itself is cloned (default).
///  - If level == -1, all the hierarchy is cloned.

REveElement* REveElement::CloneElementRecurse(Int_t level) const
{
   REveElement* el = CloneElement();
   if (level--)
   {
      CloneChildrenRecurse(el, level);
   }
   return el;
}

////////////////////////////////////////////////////////////////////////////////
/// Clone children and attach them to the dest element.
/// If level ==  0, only the direct descendants are cloned (default).
/// If level == -1, all the hierarchy is cloned.

void REveElement::CloneChildrenRecurse(REveElement* dest, Int_t level) const
{
   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      dest->AddElement((*i)->CloneElementRecurse(level));
   }
}


//==============================================================================

////////////////////////////////////////////////////////////////////////////////
/// Virtual function for retrieving name of the element.
/// Here we attempt to cast the assigned object into TNamed and call
/// GetName() there.

const char* REveElement::GetElementName() const
{
   static const REveException eh("REveElement::GetElementName ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   return named ? named->GetName() : "<no-name>";
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function for retrieving title of the render-element.
/// Here we attempt to cast the assigned object into TNamed and call
/// GetTitle() there.

const char*  REveElement::GetElementTitle() const
{
   static const REveException eh("REveElement::GetElementTitle ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   return named ? named->GetTitle() : "<no-title>";
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function for setting of name of an element.
/// Here we attempt to cast the assigned object into TNamed and call
/// SetName() there.
/// If you override this call NameTitleChanged() from there.

void REveElement::SetElementName(const char* name)
{
   static const REveException eh("REveElement::SetElementName ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   if (named) {
      named->SetName(name);
      NameTitleChanged();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function for setting of title of an element.
/// Here we attempt to cast the assigned object into TNamed and call
/// SetTitle() there.
/// If you override this call NameTitleChanged() from there.

void REveElement::SetElementTitle(const char* title)
{
   static const REveException eh("REveElement::SetElementTitle ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   if (named) {
      named->SetTitle(title);
      NameTitleChanged();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function for setting of name and title of render element.
/// Here we attempt to cast the assigned object into TNamed and call
/// SetNameTitle() there.
/// If you override this call NameTitleChanged() from there.

void REveElement::SetElementNameTitle(const char* name, const char* title)
{
   static const REveException eh("REveElement::SetElementNameTitle ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   if (named) {
      named->SetNameTitle(name, title);
      NameTitleChanged();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function called when a name or title of the element has
/// been changed.
/// If you override this, call also the version of your direct base-class.

void REveElement::NameTitleChanged()
{
   // Nothing to do - list-tree-items take this info directly.
}

//******************************************************************************

////////////////////////////////////////////////////////////////////////////////
/// Set visualization-parameter model element.
/// Calling of this function from outside of EVE should in principle
/// be avoided as it can lead to dis-synchronization of viz-tag and
/// viz-model.

void REveElement::SetVizModel(REveElement* model)
{
   if (fVizModel) {
      --fParentIgnoreCnt;
      fVizModel->RemoveElement(this);
   }
   fVizModel = model;
   if (fVizModel) {
      fVizModel->AddElement(this);
      ++fParentIgnoreCnt;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Find model element in VizDB that corresponds to previously
/// assigned fVizTag and set fVizModel accordingly.
/// If the tag is not found in VizDB, the old model-element is kept
/// and false is returned.

Bool_t REveElement::FindVizModel()
{
   REveElement* model = REX::gEve->FindVizDBEntry(fVizTag);
   if (model)
   {
      SetVizModel(model);
      return kTRUE;
   }
   else
   {
      return kFALSE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set the VizTag, find model-element from the VizDB and copy
/// visualization-parameters from it. If the model is not found and
/// fallback_tag is non-null, its search is attempted as well.
/// For example: ApplyVizTag("TPC Clusters", "Clusters");
///
/// If the model-element can not be found a warning is printed and
/// false is returned.

Bool_t REveElement::ApplyVizTag(const TString& tag, const TString& fallback_tag)
{
   SetVizTag(tag);
   if (FindVizModel())
   {
      CopyVizParamsFromDB();
      return kTRUE;
   }
   if ( ! fallback_tag.IsNull())
   {
      SetVizTag(fallback_tag);
      if (FindVizModel())
      {
         CopyVizParamsFromDB();
         return kTRUE;
      }
   }
   Warning("REveElement::ApplyVizTag", "entry for tag '%s' not found in VizDB.", tag.Data());
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate visualization parameters to dependent elements.
///
/// MainColor is propagated independently in SetMainColor().
/// In this case, as fMainColor is a pointer to Color_t, it should
/// be set in TProperClass::CopyVizParams().
///
/// Render state is not propagated. Maybe it should be, at least optionally.

void REveElement::PropagateVizParamsToProjecteds()
{
   REveProjectable* pable = dynamic_cast<REveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->PropagateVizParams();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate visualization parameters from element el (defaulting
/// to this) to all elements (children).
///
/// The primary use of this is for model-elements from
/// visualization-parameter database.

void REveElement::PropagateVizParamsToElements(REveElement* el)
{
   if (el == 0)
      el = this;

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->CopyVizParams(el);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy visualization parameters from element el.
/// This method needs to be overriden by any class that introduces
/// new parameters.
/// Color is copied in sub-classes which define it.
/// See, for example, REvePointSet::CopyVizParams(),
/// REveLine::CopyVizParams() and REveTrack::CopyVizParams().

void REveElement::CopyVizParams(const REveElement* el)
{
   fCanEditMainColor        = el->fCanEditMainColor;
   fCanEditMainTransparency = el->fCanEditMainTransparency;
   fMainTransparency        = el->fMainTransparency;

   AddStamp(kCBColorSelection | kCBObjProps);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy visualization parameters from the model-element fVizModel.
/// A warning is printed if the model-element fVizModel is not set.

void REveElement::CopyVizParamsFromDB()
{
   if (fVizModel)
   {
      CopyVizParams(fVizModel);
   }
   else
   {
      Warning("REveElement::CopyVizParamsFromDB", "VizModel has not been set.");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save visualization parameters for this element with given tag.
///
/// This function creates the instantiation code, calls virtual
/// WriteVizParams() and, at the end, writes out the code for
/// registration of the model into the VizDB.

void REveElement::SaveVizParams(std::ostream& out, const TString& tag, const TString& var)
{
   static const REveException eh("REveElement::GetObject ");

   TString t = "   ";
   TString cls(GetObject(eh)->ClassName());

   out << "\n";

   TString intro = " TAG='" + tag + "', CLASS='" + cls + "'";
   out << "   //" << intro << "\n";
   out << "   //" << TString('-', intro.Length()) << "\n";
   out << t << cls << "* " << var <<" = new " << cls << ";\n";

   WriteVizParams(out, var);

   out << t << "REX::gEve->InsertVizDBEntry(\"" << tag << "\", "<< var <<");\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Write-out visual parameters for this object.
/// This is a virtual function and all sub-classes are required to
/// first call the base-element version.
/// The name of the element pointer is 'x%03d', due to cint limitations.
/// Three spaces should be used for indentation, same as in
/// SavePrimitive() methods.

void REveElement::WriteVizParams(std::ostream& out, const TString& var)
{
   TString t = "   " + var + "->";

   out << t << "SetElementName(\""  << GetElementName()  << "\");\n";
   out << t << "SetElementTitle(\"" << GetElementTitle() << "\");\n";
   out << t << "SetEditMainColor("  << fCanEditMainColor << ");\n";
   out << t << "SetEditMainTransparency(" << fCanEditMainTransparency << ");\n";
   out << t << "SetMainTransparency("     << fMainTransparency << ");\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Set visual parameters for this object for given tag.

void REveElement::VizDB_Apply(const char* tag)
{
   if (ApplyVizTag(tag))
   {
      PropagateVizParamsToProjecteds();
      REX::gEve->Redraw3D();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Reset visual parameters for this object from VizDB.
/// The model object must be already set.

void REveElement::VizDB_Reapply()
{
   if (fVizModel)
   {
      CopyVizParamsFromDB();
      PropagateVizParamsToProjecteds();
      REX::gEve->Redraw3D();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy visual parameters from this element to viz-db model.
/// If update is set, all clients of the model will be updated to
/// the new value.
/// A warning is printed if the model-element fVizModel is not set.

void REveElement::VizDB_UpdateModel(Bool_t update)
{
   if (fVizModel)
   {
      fVizModel->CopyVizParams(this);
      if (update)
      {
         fVizModel->PropagateVizParamsToElements(fVizModel);
         REX::gEve->Redraw3D();
      }
   }
   else
   {
      Warning("VizDB_UpdateModel", "VizModel has not been set.");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Create a replica of element and insert it into VizDB with given tag.
/// If replace is true an existing element with the same tag will be replaced.
/// If update is true, existing client of tag will be updated.

void REveElement::VizDB_Insert(const char* tag, Bool_t replace, Bool_t update)
{
   static const REveException eh("REveElement::GetObject ");

   TClass* cls = GetObject(eh)->IsA();
   REveElement* el = reinterpret_cast<REveElement*>(cls->New());
   if (el == 0) {
      Error("VizDB_Insert", "Creation of replica failed.");
      return;
   }
   el->CopyVizParams(this);
   Bool_t succ = REX::gEve->InsertVizDBEntry(tag, el, replace, update);
   if (succ && update)
      REX::gEve->Redraw3D();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the master element - that is:
/// - master of projectable, if this is a projected;
/// - master of compound, if fCompound is set;
/// - master of first compound parent, if kSCBTakeAnyParentAsMaster bit is set;
/// If non of the above is true, *this* is returned.

REveElement* REveElement::GetMaster()
{
   REveProjected* proj = dynamic_cast<REveProjected*>(this);
   if (proj)
   {
      return dynamic_cast<REveElement*>(proj->GetProjectable())->GetMaster();
   }
   if (fCompound)
   {
      return fCompound->GetMaster();
   }
   if (TestCSCBits(kCSCBTakeAnyParentAsMaster))
   {
      for (List_i i = fParents.begin(); i != fParents.end(); ++i)
         if (dynamic_cast<REveCompound*>(*i))
            return (*i)->GetMaster();
   }
   return this;
}

////////////////////////////////////////////////////////////////////////////////
/// Add el into the list parents.
///
/// Adding parent is subordinate to adding an element.
/// This is an internal function.

void REveElement::AddParent(REveElement* el)
{
   assert(el != 0);

   if (fParents.empty())
   {
      fMother = el;
   }
   fParents.push_back(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove el from the list of parents.
/// Removing parent is subordinate to removing an element.
/// This is an internal function.

void REveElement::RemoveParent(REveElement* el)
{
   static const REveException eh("REveElement::RemoveParent ");

   assert(el != 0);

   if (el == fMother) fMother = 0;
   fParents.remove(el);
   CheckReferenceCount(eh);
}

/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/// Check external references to this and eventually auto-destruct
/// the render-element.

void REveElement::CheckReferenceCount(const REveException& eh)
{
   if (fDestructing != kNone)
      return;

   if (NumParents() <= fParentIgnoreCnt &&
       fDestroyOnZeroRefCnt             && fDenyDestroy <= 0)
   {
      if (REX::gEve && REX::gEve->GetUseOrphanage())
      {
         if (gDebug > 0)
            Info(eh, "moving to orphanage '%s' on zero reference count.", GetElementName());

         PreDeleteElement();
         REX::gEve->GetOrphanage()->AddElement(this);
      }
      else
      {
         if (gDebug > 0)
            Info(eh, "auto-destructing '%s' on zero reference count.", GetElementName());

         PreDeleteElement();
         delete this;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Collect all parents of class REveScene. This is needed to
/// automatically detect which scenes need to be updated.
///
/// Overriden in REveScene to include itself and return.

void REveElement::CollectSceneParents(List_t& scenes)
{
   for (List_i p=fParents.begin(); p!=fParents.end(); ++p)
      (*p)->CollectSceneParents(scenes);
}

////////////////////////////////////////////////////////////////////////////////
/// Collect scene-parents from all children. This is needed to
/// automatically detect which scenes need to be updated during/after
/// a full sub-tree update.
/// Argument parent specifies parent in traversed hierarchy for which we can
/// skip the upwards search.

void REveElement::CollectSceneParentsFromChildren(List_t&      scenes,
                                                  REveElement* parent)
{
   for (List_i p=fParents.begin(); p!=fParents.end(); ++p)
   {
      if (*p != parent) (*p)->CollectSceneParents(scenes);
   }

   for (List_i c=fChildren.begin(); c!=fChildren.end(); ++c)
   {
      (*c)->CollectSceneParentsFromChildren(scenes, this);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Get a TObject associated with this render-element.
/// Most cases uses double-inheritance from REveElement and TObject
/// so we just do a dynamic cast here.
/// If some REveElement descendant implements a different scheme,
/// this virtual method should be overriden accordingly.

TObject* REveElement::GetObject(const REveException& eh) const
{
   const TObject* obj = dynamic_cast<const TObject*>(this);
   if (obj == 0)
      throw eh + "not a TObject.";
   return const_cast<TObject*>(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Export render-element to CINT with variable name var_name.

void REveElement::ExportToCINT(char* var_name)
{
   const char* cname = IsA()->GetName();
   gROOT->ProcessLine(TString::Format("%s* %s = (%s*)0x%lx;", cname, var_name, cname, (ULong_t)this));
}

////////////////////////////////////////////////////////////////////////////////
/// Call Dump() on source object.
/// Throws an exception if it is not set.

void REveElement::DumpSourceObject() const
{
   static const REveException eh("REveElement::DumpSourceObject ");

   TObject *so = GetSourceObject();
   if (!so)
      throw eh + "source-object not set.";

   so->Dump();
}

////////////////////////////////////////////////////////////////////////////////
/// Call Print() on source object.
/// Throws an exception if it is not set.

void REveElement::PrintSourceObject() const
{
   static const REveException eh("REveElement::PrintSourceObject ");

   TObject *so = GetSourceObject();
   if (!so)
      throw eh + "source-object not set.";

   so->Print();
}

////////////////////////////////////////////////////////////////////////////////
/// Export source object to CINT with given name for the variable.
/// Throws an exception if it is not set.

void REveElement::ExportSourceObjectToCINT(char* var_name) const
{
   static const REveException eh("REveElement::ExportSourceObjectToCINT ");

   TObject *so = GetSourceObject();
   if (!so)
      throw eh + "source-object not set.";

   const char* cname = so->IsA()->GetName();
   gROOT->ProcessLine(TString::Format("%s* %s = (%s*)0x%lx;", cname, var_name, cname, (ULong_t)so));
}

/*
////////////////////////////////////////////////////////////////////////////////
/// Paint self and/or children into currently active pad.

void REveElement::PadPaint(Option_t* option)
{
   static const REveException eh("REveElement::PadPaint ");

   TObject* obj = 0;
   if (GetRnrSelf() && (obj = GetRenderObject(eh)))
      obj->Paint(option);


   if (GetRnrChildren()) {
      for (List_i i=BeginChildren(); i!=EndChildren(); ++i) {
         (*i)->PadPaint(option);
      }
   }
}
*/

 /*
////////////////////////////////////////////////////////////////////////////////
/// Paint object -- a generic implementation for EVE elements.
/// This supports direct rendering using a dedicated GL class.
/// Override TObject::Paint() in sub-classes if different behaviour
/// is required.

void REveElement::PaintStandard(TObject* id)
{
   static const REveException eh("REveElement::PaintStandard ");

   TBuffer3D buff(TBuffer3DTypes::kGeneric);

   // Section kCore
   buff.fID           = id;
   buff.fColor        = GetMainColor();
   buff.fTransparency = GetMainTransparency();
   if (HasMainTrans())  RefMainTrans().SetBuffer3D(buff);

   buff.SetSectionsValid(TBuffer3D::kCore);

   Int_t reqSections = gPad->GetViewer3D()->AddObject(buff);
   if (reqSections != TBuffer3D::kNone)
   {
      Warning(eh, "IsA='%s'. Viewer3D requires more sections (%d). Only direct-rendering supported.",
              id->ClassName(), reqSections);
   }
}
 */

////////////////////////////////////////////////////////////////////////////////
/// Set render state of this element, i.e. if it will be published
/// on next scene update pass.
/// Returns true if the state has changed.

Bool_t REveElement::SetRnrSelf(Bool_t rnr)
{
   if (SingleRnrState())
   {
      return SetRnrState(rnr);
   }

   if (rnr != fRnrSelf)
   {
      fRnrSelf = rnr;
      StampVisibility();
      PropagateRnrStateToProjecteds();

      return kTRUE;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set render state of this element's children, i.e. if they will
/// be published on next scene update pass.
/// Returns true if the state has changed.

Bool_t REveElement::SetRnrChildren(Bool_t rnr)
{
   if (SingleRnrState())
   {
      return SetRnrState(rnr);
   }

   if (rnr != fRnrChildren)
   {
      fRnrChildren = rnr;
      StampVisibility();
      PropagateRnrStateToProjecteds();
      return kTRUE;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set state for rendering of this element and its children.
/// Returns true if the state has changed.

Bool_t REveElement::SetRnrSelfChildren(Bool_t rnr_self, Bool_t rnr_children)
{
   if (SingleRnrState())
   {
      return SetRnrState(rnr_self);
   }

   if (fRnrSelf != rnr_self || fRnrChildren != rnr_children)
   {
      fRnrSelf     = rnr_self;
      fRnrChildren = rnr_children;
      StampVisibility();
      PropagateRnrStateToProjecteds();
      return kTRUE;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set render state of this element and of its children to the same
/// value.
/// Returns true if the state has changed.

Bool_t REveElement::SetRnrState(Bool_t rnr)
{
   if (fRnrSelf != rnr || fRnrChildren != rnr)
   {
      fRnrSelf = fRnrChildren = rnr;
      StampVisibility();
      PropagateRnrStateToProjecteds();
      return kTRUE;
   }
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate render state to the projected replicas of this element.
/// Maybe this should be optional on REX::gEve/element level.

void REveElement::PropagateRnrStateToProjecteds()
{
   REveProjectable *pable = dynamic_cast<REveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->PropagateRenderState(fRnrSelf, fRnrChildren);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set main color of the element.
///
///
/// List-tree-items are updated.

void REveElement::SetMainColor(Color_t color)
{
   Color_t old_color = GetMainColor();

   if (fMainColorPtr)
   {
      *fMainColorPtr = color;
      StampColorSelection();
   }

   PropagateMainColorToProjecteds(color, old_color);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert pixel to Color_t and call SetMainColor().

void REveElement::SetMainColorPixel(Pixel_t pixel)
{
   SetMainColor(TColor::GetColor(pixel));
}

////////////////////////////////////////////////////////////////////////////////
/// Convert RGB values to Color_t and call SetMainColor.

void REveElement::SetMainColorRGB(UChar_t r, UChar_t g, UChar_t b)
{
   SetMainColor(TColor::GetColor(r, g, b));
}

////////////////////////////////////////////////////////////////////////////////
/// Convert RGB values to Color_t and call SetMainColor.

void REveElement::SetMainColorRGB(Float_t r, Float_t g, Float_t b)
{
   SetMainColor(TColor::GetColor(r, g, b));
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate color to projected elements.

void REveElement::PropagateMainColorToProjecteds(Color_t color, Color_t old_color)
{
   REveProjectable* pable = dynamic_cast<REveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->PropagateMainColor(color, old_color);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set main-transparency.
/// Transparency is clamped to [0, 100].

void REveElement::SetMainTransparency(Char_t t)
{
   Char_t old_t = GetMainTransparency();

   if (t > 100) t = 100;
   fMainTransparency = t;
   StampColorSelection();

   PropagateMainTransparencyToProjecteds(t, old_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Set main-transparency via float alpha variable.
/// Value of alpha is clamped t0 [0, 1].

void REveElement::SetMainAlpha(Float_t alpha)
{
   if (alpha < 0) alpha = 0;
   if (alpha > 1) alpha = 1;
   SetMainTransparency((Char_t) (100.0f*(1.0f - alpha)));
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate transparency to projected elements.

void REveElement::PropagateMainTransparencyToProjecteds(Char_t t, Char_t old_t)
{
   REveProjectable* pable = dynamic_cast<REveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->PropagateMainTransparency(t, old_t);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return pointer to main transformation. If 'create' flag is set (default)
/// it is created if not yet existing.

REveTrans *REveElement::PtrMainTrans(Bool_t create)
{
   if (!fMainTrans && create)
      InitMainTrans();

   return fMainTrans.get();
}

////////////////////////////////////////////////////////////////////////////////
/// Return reference to main transformation. It is created if not yet
/// existing.

REveTrans &REveElement::RefMainTrans()
{
   if (!fMainTrans)
      InitMainTrans();

   return *fMainTrans.get();
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the main transformation to identity matrix.
/// If can_edit is true (default), the user will be able to edit the
/// transformation parameters via GUI.

void REveElement::InitMainTrans(Bool_t can_edit)
{
   if (fMainTrans)
      fMainTrans->UnitTrans();
   else
      fMainTrans = std::make_unique<REveTrans>();
   fCanEditMainTrans = can_edit;
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy the main transformation matrix, it will always be taken
/// as identity. Editing of transformation parameters is disabled.

void REveElement::DestroyMainTrans()
{
   fMainTrans.reset(nullptr);
   fCanEditMainTrans = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set transformation matrix from column-major array.

void REveElement::SetTransMatrix(Double_t* carr)
{
   RefMainTrans().SetFrom(carr);
}

////////////////////////////////////////////////////////////////////////////////
/// Set transformation matrix from TGeo's matrix.

void REveElement::SetTransMatrix(const TGeoMatrix& mat)
{
   RefMainTrans().SetFrom(mat);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if el can be added to this element.
///
/// In the base-class version we only make sure the new child is not
/// equal to this.

Bool_t REveElement::AcceptElement(REveElement* el)
{
   return el != this;
}

////////////////////////////////////////////////////////////////////////////////
/// Add el to the list of children.

void REveElement::AddElement(REveElement* el)
{
   static const REveException eh("REveElement::AddElement ");

   assert(el != 0);

   if ( ! AcceptElement(el))
      throw eh + Form("parent '%s' rejects '%s'.",
                      GetElementName(), el->GetElementName());

   if (el->fElementId == 0 && fElementId != 0)
   {
      el->assign_element_id_recurisvely();
   }
   if (el->fScene == 0 && fScene != 0)
   {
      el->assign_scene_recursively(fScene);
   }
   if (el->fMother == 0)
   {
      el->fMother = this;
   }

   el->AddParent(this);
   fChildren.push_back(el); ++fNumChildren;

   // XXXX This should be element added. Also, should be different for
   // "full (re)construction". Scenes should manage that and have
   // state like: none - constructing - clearing - nominal - updating.
   // I recon this means n element should have a ptr to its scene.
   // XXXXElementChanged();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove el from the list of children.

void REveElement::RemoveElement(REveElement* el)
{
   assert(el != 0);

   RemoveElementLocal(el);
   el->RemoveParent(this);
   fChildren.remove(el); --fNumChildren;
   // XXXX This should be element removed. Also, think about recursion, deletion etc.
   // XXXXElementChanged();
}

////////////////////////////////////////////////////////////////////////////////
/// Perform additional local removal of el.
/// Called from RemoveElement() which does whole untangling.
/// Put into special function as framework-related handling of
/// element removal should really be common to all classes and
/// clearing of local structures happens in between removal
/// of list-tree-items and final removal.
/// If you override this, you should also override
/// RemoveElementsLocal().

void REveElement::RemoveElementLocal(REveElement* /*el*/)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all elements. This assumes removing of all elements can
/// be done more efficiently then looping over them and removing one
/// by one. This protected function performs the removal on the
/// level of REveElement.

void REveElement::RemoveElementsInternal()
{
   // for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
   // {
   //    DestroyListSubTree(i->fTree, i->fItem);
   // }
   RemoveElementsLocal();
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->RemoveParent(this);
   }
   fChildren.clear(); fNumChildren = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all elements. This assumes removing of all elements can
/// be done more efficiently then looping over them and removing
/// them one by one.

void REveElement::RemoveElements()
{
   if (HasChildren())
   {
      RemoveElementsInternal();
      // ElementChanged();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Perform additional local removal of all elements.
/// See comment to RemoveElementlocal(REveElement*).

void REveElement::RemoveElementsLocal()
{
}

////////////////////////////////////////////////////////////////////////////////
/// If this is a projectable, loop over all projected replicas and
/// add the projected image of child 'el' there. This is supposed to
/// be called after you add a child to a projectable after it has
/// already been projected.
/// You might also want to call RecheckImpliedSelections() on this
/// element or 'el'.
///
/// If 'same_depth' flag is true, the same depth as for parent object
/// is used in every projection. Otherwise current depth of each
/// relevant projection-manager is used.

void REveElement::ProjectChild(REveElement* el, Bool_t same_depth)
{
   REveProjectable* pable = dynamic_cast<REveProjectable*>(this);
   if (pable && HasChild(el))
   {
      for (REveProjectable::ProjList_i i = pable->BeginProjecteds(); i != pable->EndProjecteds(); ++i)
      {
         REveProjectionManager *pmgr = (*i)->GetManager();
         Float_t cd = pmgr->GetCurrentDepth();
         if (same_depth) pmgr->SetCurrentDepth((*i)->GetDepth());

         pmgr->SubImportElements(el, (*i)->GetProjectedAsElement());

         if (same_depth) pmgr->SetCurrentDepth(cd);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// If this is a projectable, loop over all projected replicas and
/// add the projected image of all children there. This is supposed
/// to be called after you destroy all children and then add new
/// ones after this element has already been projected.
/// You might also want to call RecheckImpliedSelections() on this
/// element.
///
/// If 'same_depth' flag is true, the same depth as for the
/// projected element is used in every projection. Otherwise current
/// depth of each relevant projection-manager is used.

void REveElement::ProjectAllChildren(Bool_t same_depth)
{
   REveProjectable* pable = dynamic_cast<REveProjectable*>(this);
   if (pable)
   {
      for (REveProjectable::ProjList_i i = pable->BeginProjecteds(); i != pable->EndProjecteds(); ++i)
      {
         REveProjectionManager *pmgr = (*i)->GetManager();
         Float_t cd = pmgr->GetCurrentDepth();
         if (same_depth) pmgr->SetCurrentDepth((*i)->GetDepth());

         pmgr->SubImportChildren(this, (*i)->GetProjectedAsElement());

         if (same_depth) pmgr->SetCurrentDepth(cd);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Check if element el is a child of this element.

Bool_t REveElement::HasChild(REveElement* el)
{
   return (std::find(fChildren.begin(), fChildren.end(), el) != fChildren.end());
}

////////////////////////////////////////////////////////////////////////////////
/// Find the first child with given name.  If cls is specified (non
/// 0), it is also checked.
///
/// Returns 0 if not found.

REveElement* REveElement::FindChild(const TString&  name, const TClass* cls)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if (name.CompareTo((*i)->GetElementName()) == 0)
      {
         if (!cls || (cls && (*i)->IsA()->InheritsFrom(cls)))
            return *i;
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Find the first child whose name matches regexp. If cls is
/// specified (non 0), it is also checked.
///
/// Returns 0 if not found.

REveElement* REveElement::FindChild(TPRegexp& regexp, const TClass* cls)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if (regexp.MatchB((*i)->GetElementName()))
      {
         if (!cls || (cls && (*i)->IsA()->InheritsFrom(cls)))
            return *i;
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Find all children with given name and append them to matches
/// list. If class is specified (non 0), it is also checked.
///
/// Returns number of elements added to the list.

Int_t REveElement::FindChildren(List_t& matches,
                                const TString& name, const TClass* cls)
{
   Int_t count = 0;
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if (name.CompareTo((*i)->GetElementName()) == 0)
      {
         if (!cls || (cls && (*i)->IsA()->InheritsFrom(cls)))
         {
            matches.push_back(*i);
            ++count;
         }
      }
   }
   return count;
}

////////////////////////////////////////////////////////////////////////////////
/// Find all children whose name matches regexp and append them to
/// matches list.
///
/// Returns number of elements added to the list.

Int_t REveElement::FindChildren(List_t& matches,
                                TPRegexp& regexp, const TClass* cls)
{
   Int_t count = 0;
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      if (regexp.MatchB((*i)->GetElementName()))
      {
         if (!cls || (cls && (*i)->IsA()->InheritsFrom(cls)))
         {
            matches.push_back(*i);
            ++count;
         }
      }
   }
   return count;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the first child element or 0 if the list is empty.

REveElement* REveElement::FirstChild() const
{
   return HasChildren() ? fChildren.front() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the last child element or 0 if the list is empty.

REveElement* REveElement::LastChild () const
{
   return HasChildren() ? fChildren.back() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Enable rendering of children and their list contents.
/// Arguments control how to set self/child rendering.

void REveElement::EnableListElements(Bool_t rnr_self,  Bool_t rnr_children)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->SetRnrSelfChildren(rnr_self, rnr_children);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Disable rendering of children and their list contents.
/// Arguments control how to set self/child rendering.
///
/// Same as above function, but default arguments are different. This
/// is convenient for calls via context menu.

void REveElement::DisableListElements(Bool_t rnr_self,  Bool_t rnr_children)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->SetRnrSelfChildren(rnr_self, rnr_children);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Protected member function called from REveElement::Annihilate().

void REveElement::AnnihilateRecursively()
{
   static const REveException eh("REveElement::AnnihilateRecursively ");

   // projected  were already destroyed in REveElement::Anihilate(), now only clear its list
   REveProjectable* pable = dynamic_cast<REveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->ClearProjectedList();
   }

   // same as REveElement::RemoveElementsInternal(), except parents are ignored
   // for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
   // {
   //    DestroyListSubTree(i->fTree, i->fItem);
   // }
   RemoveElementsLocal();
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->AnnihilateRecursively();
   }

   fChildren.clear();
   fNumChildren = 0;

   fDestructing = kAnnihilate;
   PreDeleteElement();

   delete this;
}

////////////////////////////////////////////////////////////////////////////////
/// Optimized destruction without check of reference-count.
/// Parents are not notified about child destruction.
/// The method should only be used when an element does not have
/// more than one parent -- otherwise an exception is thrown.

void REveElement::Annihilate()
{
   static const REveException eh("REveElement::Annihilate ");

   if (fParents.size() > 1)
   {
      Warning(eh, "More than one parent for '%s': %d. Refusing to delete.",
              GetElementName(), (Int_t) fParents.size());
      return;
   }

   fDestructing = kAnnihilate;

   // recursive annihilation of projecteds
   REveProjectable* pable = dynamic_cast<REveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->AnnihilateProjecteds();
   }

   // detach from the parent
   while (!fParents.empty())
   {
      fParents.front()->RemoveElement(this);
   }

   AnnihilateRecursively();

   REX::gEve->Redraw3D();
}

////////////////////////////////////////////////////////////////////////////////
/// Annihilate elements.

void REveElement::AnnihilateElements()
{
   while (!fChildren.empty())
   {
      REveElement* c = fChildren.front();
      c->Annihilate();
   }

   fNumChildren = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy this element. Throws an exception if deny-destroy is in force.
/// This method should be called instead of a destructor.
/// Note that an exception will be thrown if the element has been
/// protected against destruction with IncDenyDestroy().

void REveElement::Destroy()
{
   static const REveException eh("REveElement::Destroy ");

   if (fDenyDestroy > 0)
      throw eh + TString::Format("element '%s' (%s*) 0x%lx is protected against destruction.",
                                 GetElementName(), IsA()->GetName(), (ULong_t)this);

   PreDeleteElement();
   delete this;
   REX::gEve->Redraw3D();
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy this element. Prints a warning if deny-destroy is in force.

void REveElement::DestroyOrWarn()
{
   static const REveException eh("REveElement::DestroyOrWarn ");

   try
   {
      Destroy();
   }
   catch (REveException& exc)
   {
      Warning(eh, "%s", exc.Data());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy all children of this element.

void REveElement::DestroyElements()
{
   static const REveException eh("REveElement::DestroyElements ");

   while (HasChildren())
   {
      REveElement* c = fChildren.front();
      if (c->fDenyDestroy <= 0)
      {
         try {
            c->Destroy();
         }
         catch (REveException& exc) {
            Warning(eh, "element destruction failed: '%s'.", exc.Data());
            RemoveElement(c);
         }
      }
      else
      {
         if (gDebug > 0)
            Info(eh, "element '%s' is protected agains destruction, removing locally.", c->GetElementName());
         RemoveElement(c);
      }
   }

   REX::gEve->Redraw3D();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns state of flag determining if the element will be
/// destroyed when reference count reaches zero.
/// This is true by default.

Bool_t REveElement::GetDestroyOnZeroRefCnt() const
{
   return fDestroyOnZeroRefCnt;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the state of flag determining if the element will be
/// destroyed when reference count reaches zero.
/// This is true by default.

void REveElement::SetDestroyOnZeroRefCnt(Bool_t d)
{
   fDestroyOnZeroRefCnt = d;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the number of times deny-destroy has been requested on
/// the element.

Int_t REveElement::GetDenyDestroy() const
{
   return fDenyDestroy;
}

////////////////////////////////////////////////////////////////////////////////
/// Increases the deny-destroy count of the element.
/// Call this if you store an external pointer to the element.

void REveElement::IncDenyDestroy()
{
   ++fDenyDestroy;
}

////////////////////////////////////////////////////////////////////////////////
/// Decreases the deny-destroy count of the element.
/// Call this after releasing an external pointer to the element.

void REveElement::DecDenyDestroy()
{
   if (--fDenyDestroy <= 0)
      CheckReferenceCount("REveElement::DecDenyDestroy ");
}

////////////////////////////////////////////////////////////////////////////////
/// Get number of parents that should be ignored in doing
/// reference-counting.
///
/// For example, this is used when subscribing an element to a
/// visualization-database model object.

Int_t REveElement::GetParentIgnoreCnt() const
{
   return fParentIgnoreCnt;
}

////////////////////////////////////////////////////////////////////////////////
/// Increase number of parents ignored in reference-counting.

void REveElement::IncParentIgnoreCnt()
{
   ++fParentIgnoreCnt;
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease number of parents ignored in reference-counting.

void REveElement::DecParentIgnoreCnt()
{
   if (--fParentIgnoreCnt <= 0)
      CheckReferenceCount("REveElement::DecParentIgnoreCnt ");
}

////////////////////////////////////////////////////////////////////////////////
/// React to element being pasted or dnd-ed.
/// Return true if redraw is needed.

Bool_t REveElement::HandleElementPaste(REveElement* el)
{
   REX::gEve->AddElement(el, this);
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Call this after an element has been changed so that the state
/// can be propagated around the framework.

void REveElement::ElementChanged(Bool_t update_scenes, Bool_t redraw)
{
   REX::gEve->ElementChanged(this, update_scenes, redraw);
}

////////////////////////////////////////////////////////////////////////////////
/// Set pickable state on the element and all its children.

void REveElement::SetPickableRecursively(Bool_t p)
{
   fPickable = p;
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->SetPickableRecursively(p);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns element to be selected on click.
/// If value is zero the selected object will follow rules in
/// REveSelection.

REveElement* REveElement::ForwardSelection()
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns element to be displayed in GUI editor on click.
/// If value is zero the displayed object will follow rules in
/// REveSelection.

REveElement* REveElement::ForwardEdit()
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set element's selection state. Stamp appropriately.

void REveElement::SelectElement(Bool_t state)
{
   if (fSelected != state) {
      fSelected = state;
      if (!fSelected && fImpliedSelected == 0)
         UnSelected();
      fParentIgnoreCnt += (fSelected) ? 1 : -1;
      StampColorSelection();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Increase element's implied-selection count. Stamp appropriately.

void REveElement::IncImpliedSelected()
{
   if (fImpliedSelected++ == 0)
      StampColorSelection();
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease element's implied-selection count. Stamp appropriately.

void REveElement::DecImpliedSelected()
{
   if (--fImpliedSelected == 0)
   {
      if (!fSelected)
         UnSelected();
      StampColorSelection();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function called when both fSelected is false and
/// fImpliedSelected is 0.
/// Nothing is done in this base-class version

void REveElement::UnSelected()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set element's highlight state. Stamp appropriately.

void REveElement::HighlightElement(Bool_t state)
{
   if (fHighlighted != state) {
      fHighlighted = state;
      if (!fHighlighted && fImpliedHighlighted == 0)
         UnHighlighted();
      fParentIgnoreCnt += (fHighlighted) ? 1 : -1;
      StampColorSelection();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Increase element's implied-highlight count. Stamp appropriately.

void REveElement::IncImpliedHighlighted()
{
   if (fImpliedHighlighted++ == 0)
      StampColorSelection();
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease element's implied-highlight count. Stamp appropriately.

void REveElement::DecImpliedHighlighted()
{
   if (--fImpliedHighlighted == 0)
   {
      if (!fHighlighted)
         UnHighlighted();
      StampColorSelection();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function called when both fHighlighted is false and
/// fImpliedHighlighted is 0.
/// Nothing is done in this base-class version

void REveElement::UnHighlighted()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Populate set impSelSet with derived / dependant elements.
///
/// If this is a REveProjectable, the projected replicas are added
/// to the set. Thus it does not have to be reimplemented for each
/// sub-class of REveProjected.
///
/// Note that this also takes care of projections of REveCompound
/// class, which is also a projectable.

void REveElement::FillImpliedSelectedSet(Set_t& impSelSet)
{
   REveProjectable* p = dynamic_cast<REveProjectable*>(this);
   if (p)
   {
      p->AddProjectedsToSet(impSelSet);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get selection level, needed for rendering selection and
/// highlight feedback.
/// This should go to TAtt3D.

UChar_t REveElement::GetSelectedLevel() const
{
   if (fSelected)               return 1;
   if (fImpliedSelected > 0)    return 2;
   if (fHighlighted)            return 3;
   if (fImpliedHighlighted > 0) return 4;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Call this if it is possible that implied-selection or highlight
/// has changed for this element or for implied-selection this
/// element is member of and you want to maintain consistent
/// selection state.
/// This can happen if you add elements into compounds in response
/// to user-interaction.

void REveElement::RecheckImpliedSelections()
{
   if (fSelected || fImpliedSelected)
      REX::gEve->GetSelection()->RecheckImpliedSetForElement(this);

   if (fHighlighted || fImpliedHighlighted)
      REX::gEve->GetHighlight()->RecheckImpliedSetForElement(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Add (bitwise or) given stamps to fChangeBits.
/// Register this element to REX::gEve as stamped.
/// This method is virtual so that sub-classes can add additional
/// actions. The base-class method should still be called (or replicated).

void REveElement::AddStamp(UChar_t bits)
{
  if (fDestructing == kNone && fScene && fScene->IsAcceptingChanges())
   {
      printf("%s AddStamp %d + (%d) -> %d \n", GetElementName(), fChangeBits, bits, fChangeBits|bits);
      fChangeBits |= bits;
      fScene->SceneElementChanged(this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Convert Bool_t to string - kTRUE or kFALSE.
/// Needed in WriteVizParams().

const char* REveElement::ToString(Bool_t b)
{
   return b ? "kTRUE" : "kFALSE";
}

/** \class REveElementObjectPtr
\ingroup REve
REveElement with external TObject as a holder of visualization data.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveElementObjectPtr::REveElementObjectPtr(TObject* obj, Bool_t own) :
   REveElement (),
   TObject     (),
   fObject     (obj),
   fOwnObject  (own)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveElementObjectPtr::REveElementObjectPtr(TObject* obj, Color_t& mainColor, Bool_t own) :
   REveElement (mainColor),
   TObject     (),
   fObject     (obj),
   fOwnObject  (own)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.
/// If object pointed to is owned it is cloned.
/// It is assumed that the main-color has its origin in the TObject pointed to so
/// it is fixed here accordingly.

REveElementObjectPtr::REveElementObjectPtr(const REveElementObjectPtr& e) :
   REveElement (e),
   TObject     (e),
   fObject     (0),
   fOwnObject  (e.fOwnObject)
{
   if (fOwnObject && e.fObject)
   {
      fObject = e.fObject->Clone();
      SetMainColorPtr((Color_t*)((const char*) fObject + ((const char*) e.GetMainColorPtr() - (const char*) e.fObject)));
   }
   else
   {
      SetMainColorPtr(e.GetMainColorPtr());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Clone the element via copy constructor.
/// Virtual from REveElement.

REveElementObjectPtr* REveElementObjectPtr::CloneElement() const
{
   return new REveElementObjectPtr(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Return external object.
/// Virtual from REveElement.

TObject* REveElementObjectPtr::GetObject(const REveException& eh) const
{
   if (fObject == 0)
      throw eh + "fObject not set.";
   return fObject;
}

////////////////////////////////////////////////////////////////////////////////
/// Export external object to CINT with variable name var_name.
/// Virtual from REveElement.

void REveElementObjectPtr::ExportToCINT(char* var_name)
{
   static const REveException eh("REveElementObjectPtr::ExportToCINT ");

   TObject* obj = GetObject(eh);
   const char* cname = obj->IsA()->GetName();
   gROOT->ProcessLine(Form("%s* %s = (%s*)0x%lx;", cname, var_name, cname, (ULong_t)obj));
}

//==============================================================================
// Write core json. If rnr_offset negative, render data will not be written
//==============================================================================

Int_t REveElement::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   j["_typename"]  = IsA()->GetName();
   j["fName"]      = GetElementName();
   j["fTitle"]     = GetElementTitle();
   j["fElementId"] = GetElementId();
   j["fMotherId"]  = get_mother_id();
   j["fSceneId"]   = get_scene_id();
   j["fMasterId"]  = GetMaster()->GetElementId();

   j["fRnrSelf"]     = GetRnrSelf();
   j["fRnrChildren"] = GetRnrChildren();

   j["fMainColor"]        = GetMainColor();
   j["fMainTransparency"] = GetMainTransparency();

   if (rnr_offset >=0) {
      BuildRenderData();
      if (fRenderData.get())
      {
         nlohmann::json rd = {};

         rd["rnr_offset"] = rnr_offset;
         rd["rnr_func"]   = fRenderData->GetRnrFunc();
         rd["vert_size"]  = fRenderData->SizeV();
         rd["norm_size"]  = fRenderData->SizeN();
         rd["index_size"] = fRenderData->SizeI();

         j["render_data"] = rd;

         return fRenderData->GetBinarySize();
      }
      else
      {
         return 0;
      }
   }
   else {
      return 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

REveElementObjectPtr::~REveElementObjectPtr()
{
   if (fOwnObject)
      delete fObject;
}

/** \class  REveElementList
\ingroup REve
A list of EveElements.

Class of acceptable children can be limited by setting the
fChildClass member.

!!! should have two ctors (like in REveElement), one with Color_t&
and set fDoColor automatically, based on which ctor is called.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveElementList::REveElementList(const char* n, const char* t, Bool_t doColor, Bool_t doTransparency) :
   REveElement(),
   TNamed(n, t),
   REveProjectable(),
   fColor(0),
   fChildClass(nullptr)
{
   if (doColor) {
      fCanEditMainColor = kTRUE;
      SetMainColorPtr(&fColor);
   }
   if (doTransparency)
   {
      fCanEditMainTransparency = kTRUE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

REveElementList::REveElementList(const REveElementList& e) :
   REveElement (e),
   TNamed      (e),
   REveProjectable(),
   fColor      (e.fColor),
   fChildClass (e.fChildClass)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Clone the element via copy constructor.
/// Virtual from REveElement.

REveElementList* REveElementList::CloneElement() const
{
   return new REveElementList(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if REveElement el is inherited from fChildClass.
/// Virtual from REveElement.

Bool_t REveElementList::AcceptElement(REveElement* el)
{
   if (fChildClass && ! el->IsA()->InheritsFrom(fChildClass))
      return kFALSE;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from REveProjectable, returns REveCompoundProjected class.

TClass* REveElementList::ProjectedClass(const REveProjection*) const
{
   return REveElementListProjected::Class();
}

/** \class REveElementListProjected
\ingroup REve
A projected element list -- required for proper propagation
of render state to projected views.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveElementListProjected::REveElementListProjected() :
   REveElementList("REveElementListProjected")
{
}

////////////////////////////////////////////////////////////////////////////////
/// This is abstract method from base-class REveProjected.
/// No implementation.

void REveElementListProjected::UpdateProjection()
{
}
