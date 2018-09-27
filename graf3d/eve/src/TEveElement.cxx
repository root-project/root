// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveElement.h"
#include "TEveCompound.h"
#include "TEveTrans.h"
#include "TEveManager.h"
#include "TEveSelection.h"
#include "TEveProjectionBases.h"
#include "TEveProjectionManager.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"

#include "TGeoMatrix.h"

#include "TClass.h"
#include "TPRegexp.h"
#include "TROOT.h"
#include "TColor.h"
#include "TEveBrowser.h"
#include "TGListTree.h"
#include "TGPicture.h"

#include <algorithm>

/** \class TEveElement::TEveListTreeInfo
\ingroup TEve
Structure holding information about TGListTree and TGListTreeItem
that represents given TEveElement. This needed because each element
can appear in several list-trees as well as several times in the
same list-tree.
*/

ClassImp(TEveElement::TEveListTreeInfo);

/** \class TEveElement
\ingroup TEve
Base class for TEveUtil visualization elements, providing hierarchy
management, rendering control and list-tree item management.
*/

ClassImp(TEveElement);

const TGPicture* TEveElement::fgRnrIcons[4]      = { 0 };
const TGPicture* TEveElement::fgListTreeIcons[9] = { 0 };

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TEveElement::TEveElement() :
   fParents             (),
   fChildren            (),
   fCompound            (0),
   fVizModel            (0),
   fVizTag              (),
   fNumChildren         (0),
   fParentIgnoreCnt     (0),
   fTopItemCnt          (0),
   fDenyDestroy         (0),
   fDestroyOnZeroRefCnt (kTRUE),
   fRnrSelf             (kTRUE),
   fRnrChildren         (kTRUE),
   fCanEditMainColor    (kFALSE),
   fCanEditMainTransparency(kFALSE),
   fCanEditMainTrans    (kFALSE),
   fMainTransparency    (0),
   fMainColorPtr        (0),
   fMainTrans           (0),
   fItems               (),
   fSource              (),
   fUserData            (0),
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

TEveElement::TEveElement(Color_t& main_color) :
   fParents             (),
   fChildren            (),
   fCompound            (0),
   fVizModel            (0),
   fVizTag              (),
   fNumChildren         (0),
   fParentIgnoreCnt     (0),
   fTopItemCnt          (0),
   fDenyDestroy         (0),
   fDestroyOnZeroRefCnt (kTRUE),
   fRnrSelf             (kTRUE),
   fRnrChildren         (kTRUE),
   fCanEditMainColor    (kFALSE),
   fCanEditMainTransparency(kFALSE),
   fCanEditMainTrans    (kFALSE),
   fMainTransparency    (0),
   fMainColorPtr        (&main_color),
   fMainTrans           (0),
   fItems               (),
   fSource              (),
   fUserData            (0),
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
///   TEveElement* CloneElementRecurse(Int_t level)
///   void         CloneChildrenRecurse(TEveElement* dest, Int_t level)
/// ~~~
/// 'TRef fSource' is copied but 'void* UserData' is NOT.
/// If the element is projectable, its projections are NOT copied.
///
/// Not implemented for most sub-classes, let us know.
/// Note that sub-classes of TEveProjected are NOT and will NOT be copyable.

TEveElement::TEveElement(const TEveElement& e) :
   fParents             (),
   fChildren            (),
   fCompound            (0),
   fVizModel            (0),
   fVizTag              (e.fVizTag),
   fNumChildren         (0),
   fParentIgnoreCnt     (0),
   fTopItemCnt          (0),
   fDenyDestroy         (0),
   fDestroyOnZeroRefCnt (e.fDestroyOnZeroRefCnt),
   fRnrSelf             (e.fRnrSelf),
   fRnrChildren         (e.fRnrChildren),
   fCanEditMainColor    (e.fCanEditMainColor),
   fCanEditMainTransparency(e.fCanEditMainTransparency),
   fCanEditMainTrans    (e.fCanEditMainTrans),
   fMainTransparency    (e.fMainTransparency),
   fMainColorPtr        (0),
   fMainTrans           (0),
   fItems               (),
   fSource              (e.fSource),
   fUserData            (0),
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
   if (e.fMainColorPtr)
      fMainColorPtr = (Color_t*)((const char*) this + ((const char*) e.fMainColorPtr - (const char*) &e));
   if (e.fMainTrans)
      fMainTrans = new TEveTrans(*e.fMainTrans);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor. Do not call this method directly, either call Destroy() or
/// Annihilate(). See also DestroyElements() and AnnihilateElements() if you
/// need to delete all children of an element.

TEveElement::~TEveElement()
{
   if (fDestructing != kAnnihilate)
   {
      fDestructing = kStandard;
      RemoveElementsInternal();

      for (List_i p=fParents.begin(); p!=fParents.end(); ++p)
      {
         (*p)->RemoveElementLocal(this);
         (*p)->fChildren.remove(this);
         --((*p)->fNumChildren);
      }
   }

   fParents.clear();

   for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
      i->fTree->DeleteItem(i->fItem);

   delete fMainTrans;
}

////////////////////////////////////////////////////////////////////////////////
/// Called before the element is deleted, thus offering the last chance
/// to detach from acquired resources and from the framework itself.
/// Here the request is just passed to TEveManager.
/// If you override it, make sure to call base-class version.

void TEveElement::PreDeleteElement()
{
   gEve->PreDeleteElement(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Clone the element via copy constructor.
/// Should be implemented for all classes that require cloning support.

TEveElement* TEveElement::CloneElement() const
{
   return new TEveElement(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Clone elements and recurse 'level' deep over children.
///  - If level ==  0, only the element itself is cloned (default).
///  - If level == -1, all the hierarchy is cloned.

TEveElement* TEveElement::CloneElementRecurse(Int_t level) const
{
   TEveElement* el = CloneElement();
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

void TEveElement::CloneChildrenRecurse(TEveElement* dest, Int_t level) const
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

const char* TEveElement::GetElementName() const
{
   static const TEveException eh("TEveElement::GetElementName ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   return named ? named->GetName() : "<no-name>";
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function for retrieving title of the render-element.
/// Here we attempt to cast the assigned object into TNamed and call
/// GetTitle() there.

const char*  TEveElement::GetElementTitle() const
{
   static const TEveException eh("TEveElement::GetElementTitle ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   return named ? named->GetTitle() : "<no-title>";
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual function for setting of name of an element.
/// Here we attempt to cast the assigned object into TNamed and call
/// SetName() there.
/// If you override this call NameTitleChanged() from there.

void TEveElement::SetElementName(const char* name)
{
   static const TEveException eh("TEveElement::SetElementName ");

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

void TEveElement::SetElementTitle(const char* title)
{
   static const TEveException eh("TEveElement::SetElementTitle ");

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

void TEveElement::SetElementNameTitle(const char* name, const char* title)
{
   static const TEveException eh("TEveElement::SetElementNameTitle ");

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

void TEveElement::NameTitleChanged()
{
   // Nothing to do - list-tree-items take this info directly.
}

//******************************************************************************

////////////////////////////////////////////////////////////////////////////////
/// Set visualization-parameter model element.
/// Calling of this function from outside of EVE should in principle
/// be avoided as it can lead to dis-synchronization of viz-tag and
/// viz-model.

void TEveElement::SetVizModel(TEveElement* model)
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

Bool_t TEveElement::FindVizModel()
{
   TEveElement* model = gEve->FindVizDBEntry(fVizTag);
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

Bool_t TEveElement::ApplyVizTag(const TString& tag, const TString& fallback_tag)
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
   Warning("TEveElement::ApplyVizTag", "entry for tag '%s' not found in VizDB.", tag.Data());
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

void TEveElement::PropagateVizParamsToProjecteds()
{
   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(this);
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

void TEveElement::PropagateVizParamsToElements(TEveElement* el)
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
/// See, for example, TEvePointSet::CopyVizParams(),
/// TEveLine::CopyVizParams() and TEveTrack::CopyVizParams().

void TEveElement::CopyVizParams(const TEveElement* el)
{
   fCanEditMainColor        = el->fCanEditMainColor;
   fCanEditMainTransparency = el->fCanEditMainTransparency;
   fMainTransparency        = el->fMainTransparency;

   AddStamp(kCBColorSelection | kCBObjProps);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy visualization parameters from the model-element fVizModel.
/// A warning is printed if the model-element fVizModel is not set.

void TEveElement::CopyVizParamsFromDB()
{
   if (fVizModel)
   {
      CopyVizParams(fVizModel);
   }
   else
   {
      Warning("TEveElement::CopyVizParamsFromDB", "VizModel has not been set.");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save visualization parameters for this element with given tag.
///
/// This function creates the instantiation code, calls virtual
/// WriteVizParams() and, at the end, writes out the code for
/// registration of the model into the VizDB.

void TEveElement::SaveVizParams(std::ostream& out, const TString& tag, const TString& var)
{
   static const TEveException eh("TEveElement::GetObject ");

   TString t = "   ";
   TString cls(GetObject(eh)->ClassName());

   out << "\n";

   TString intro = " TAG='" + tag + "', CLASS='" + cls + "'";
   out << "   //" << intro << "\n";
   out << "   //" << TString('-', intro.Length()) << "\n";
   out << t << cls << "* " << var <<" = new " << cls << ";\n";

   WriteVizParams(out, var);

   out << t << "gEve->InsertVizDBEntry(\"" << tag << "\", "<< var <<");\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Write-out visual parameters for this object.
/// This is a virtual function and all sub-classes are required to
/// first call the base-element version.
/// The name of the element pointer is 'x%03d', due to cint limitations.
/// Three spaces should be used for indentation, same as in
/// SavePrimitive() methods.

void TEveElement::WriteVizParams(std::ostream& out, const TString& var)
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

void TEveElement::VizDB_Apply(const char* tag)
{
   if (ApplyVizTag(tag))
   {
      PropagateVizParamsToProjecteds();
      gEve->Redraw3D();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Reset visual parameters for this object from VizDB.
/// The model object must be already set.

void TEveElement::VizDB_Reapply()
{
   if (fVizModel)
   {
      CopyVizParamsFromDB();
      PropagateVizParamsToProjecteds();
      gEve->Redraw3D();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Copy visual parameters from this element to viz-db model.
/// If update is set, all clients of the model will be updated to
/// the new value.
/// A warning is printed if the model-element fVizModel is not set.

void TEveElement::VizDB_UpdateModel(Bool_t update)
{
   if (fVizModel)
   {
      fVizModel->CopyVizParams(this);
      if (update)
      {
         fVizModel->PropagateVizParamsToElements(fVizModel);
         gEve->Redraw3D();
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

void TEveElement::VizDB_Insert(const char* tag, Bool_t replace, Bool_t update)
{
   static const TEveException eh("TEveElement::GetObject ");

   TClass* cls = GetObject(eh)->IsA();
   TEveElement* el = reinterpret_cast<TEveElement*>(cls->New());
   if (el == 0) {
      Error("VizDB_Insert", "Creation of replica failed.");
      return;
   }
   el->CopyVizParams(this);
   Bool_t succ = gEve->InsertVizDBEntry(tag, el, replace, update);
   if (succ && update)
      gEve->Redraw3D();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the master element - that is:
/// - master of projectable, if this is a projected;
/// - master of compound, if fCompound is set;
/// - master of first compound parent, if kSCBTakeAnyParentAsMaster bit is set;
/// If non of the above is true, *this* is returned.

TEveElement* TEveElement::GetMaster()
{
   TEveProjected* proj = dynamic_cast<TEveProjected*>(this);
   if (proj)
   {
      return dynamic_cast<TEveElement*>(proj->GetProjectable())->GetMaster();
   }
   if (fCompound)
   {
      return fCompound->GetMaster();
   }
   if (TestCSCBits(kCSCBTakeAnyParentAsMaster))
   {
      for (List_i i = fParents.begin(); i != fParents.end(); ++i)
         if (dynamic_cast<TEveCompound*>(*i))
            return (*i)->GetMaster();
   }
   return this;
}

////////////////////////////////////////////////////////////////////////////////
/// Add re into the list parents.
///
/// Adding parent is subordinate to adding an element.
/// This is an internal function.

void TEveElement::AddParent(TEveElement* re)
{
   fParents.push_back(re);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove re from the list of parents.
/// Removing parent is subordinate to removing an element.
/// This is an internal function.

void TEveElement::RemoveParent(TEveElement* re)
{
   static const TEveException eh("TEveElement::RemoveParent ");

   fParents.remove(re);
   CheckReferenceCount(eh);
}

/******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
/// Check external references to this and eventually auto-destruct
/// the render-element.

void TEveElement::CheckReferenceCount(const TEveException& eh)
{
   if (fDestructing != kNone)
      return;

   if (NumParents() <= fParentIgnoreCnt && fTopItemCnt  <= 0 &&
       fDestroyOnZeroRefCnt             && fDenyDestroy <= 0)
   {
      if (gEve->GetUseOrphanage())
      {
         if (gDebug > 0)
            Info(eh, "moving to orphanage '%s' on zero reference count.", GetElementName());

         PreDeleteElement();
         gEve->GetOrphanage()->AddElement(this);
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
/// Collect all parents of class TEveScene. This is needed to
/// automatically detect which scenes need to be updated.
///
/// Overriden in TEveScene to include itself and return.

void TEveElement::CollectSceneParents(List_t& scenes)
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

void TEveElement::CollectSceneParentsFromChildren(List_t&      scenes,
                                                  TEveElement* parent)
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
/// Populates parent with elements.
/// parent must be an already existing representation of *this*.
/// Returns number of inserted elements.
/// If parent already has children, it does nothing.
///
/// Element can be inserted in a list-tree several times, thus we can not
/// search through fItems to get parent here.
/// Anyhow, it is probably known as it must have been selected by the user.

void TEveElement::ExpandIntoListTree(TGListTree* ltree,
                                     TGListTreeItem* parent)
{
   if (parent->GetFirstChild() != 0)
      return;
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i) {
      (*i)->AddIntoListTree(ltree, parent);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy sub-tree under item 'parent' in list-tree 'ltree'.

void TEveElement::DestroyListSubTree(TGListTree* ltree,
                                     TGListTreeItem* parent)
{
   TGListTreeItem* i = parent->GetFirstChild();
   while (i != 0)
   {
      TEveElement* re = (TEveElement*) i->GetUserData();
      i = i->GetNextSibling();
      re->RemoveFromListTree(ltree, parent);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add this element into ltree to an already existing item
/// parent_lti. Children, if any, are added as below the newly created item.
/// Returns the newly created list-tree-item.

TGListTreeItem* TEveElement::AddIntoListTree(TGListTree* ltree,
                                             TGListTreeItem* parent_lti)
{
   static const TEveException eh("TEveElement::AddIntoListTree ");

   TGListTreeItem* item = new TEveListTreeItem(this);
   ltree->AddItem(parent_lti, item);
   fItems.insert(TEveListTreeInfo(ltree, item));

   if (parent_lti == 0)
      ++fTopItemCnt;

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->AddIntoListTree(ltree, item);
   }

   ltree->ClearViewPort();

   return item;
}

////////////////////////////////////////////////////////////////////////////////
/// Add this render element into ltree to all items belonging to
/// parent. Returns list-tree-item from the first register entry (but
/// we use a set for that so it can be anything).

TGListTreeItem* TEveElement::AddIntoListTree(TGListTree* ltree,
                                             TEveElement* parent)
{
   TGListTreeItem* lti = 0;
   if (parent == 0) {
      lti = AddIntoListTree(ltree, (TGListTreeItem*) 0);
   } else {
      for (sLTI_ri i = parent->fItems.rbegin(); i != parent->fItems.rend(); ++i)
      {
         if (i->fTree == ltree)
            lti = AddIntoListTree(ltree, i->fItem);
      }
   }
   return lti;
}

////////////////////////////////////////////////////////////////////////////////
/// Add this render element into all list-trees and all items
/// belonging to parent. Returns list-tree-item from the first
/// register entry (but we use a set for that so it can be anything).

TGListTreeItem* TEveElement::AddIntoListTrees(TEveElement* parent)
{
   TGListTreeItem* lti = 0;
   for (sLTI_ri i = parent->fItems.rbegin(); i != parent->fItems.rend(); ++i)
   {
      lti = AddIntoListTree(i->fTree, i->fItem);
   }
   return lti;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove element from list-tree 'ltree' where its parent item is
/// 'parent_lti'.
/// Returns kTRUE if the item was found and removed, kFALSE
/// otherwise.

Bool_t TEveElement::RemoveFromListTree(TGListTree* ltree,
                                       TGListTreeItem* parent_lti)
{
   static const TEveException eh("TEveElement::RemoveFromListTree ");

   sLTI_i i = FindItem(ltree, parent_lti);
   if (i != fItems.end()) {
      DestroyListSubTree(ltree, i->fItem);
      ltree->DeleteItem(i->fItem);
      ltree->ClearViewPort();
      fItems.erase(i);
      if (parent_lti == 0) {
         --fTopItemCnt;
         CheckReferenceCount(eh);
      }
      return kTRUE;
   } else {
      return kFALSE;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Remove element from all list-trees where 'parent' is the
/// user-data of the parent list-tree-item.

Int_t TEveElement::RemoveFromListTrees(TEveElement* parent)
{
   static const TEveException eh("TEveElement::RemoveFromListTrees ");

   Int_t count = 0;

   sLTI_i i  = fItems.begin();
   while (i != fItems.end())
   {
      sLTI_i j = i++;
      TGListTreeItem *plti = j->fItem->GetParent();
      if ((plti != 0 && (TEveElement*) plti->GetUserData() == parent) ||
          (plti == 0 && parent == 0))
      {
         DestroyListSubTree(j->fTree, j->fItem);
         j->fTree->DeleteItem(j->fItem);
         j->fTree->ClearViewPort();
         fItems.erase(j);
         if (parent == 0)
            --fTopItemCnt;
         ++count;
      }
   }

   if (parent == 0 && count > 0)
      CheckReferenceCount(eh);

   return count;
}

////////////////////////////////////////////////////////////////////////////////
/// Find any list-tree-item of this element in list-tree 'ltree'.
/// Note that each element can be placed into the same list-tree on
/// several postions.

TEveElement::sLTI_i TEveElement::FindItem(TGListTree* ltree)
{
   for (sLTI_i i = fItems.begin(); i != fItems.end(); ++i)
      if (i->fTree == ltree)
         return i;
   return fItems.end();
}

////////////////////////////////////////////////////////////////////////////////
/// Find list-tree-item of this element with given parent
/// list-tree-item.

TEveElement::sLTI_i TEveElement::FindItem(TGListTree* ltree,
                                          TGListTreeItem* parent_lti)
{
   for (sLTI_i i = fItems.begin(); i != fItems.end(); ++i)
      if (i->fTree == ltree && i->fItem->GetParent() == parent_lti)
         return i;
   return fItems.end();
}

////////////////////////////////////////////////////////////////////////////////
/// Find any list-tree-item of this element in list-tree 'ltree'.
/// Note that each element can be placed into the same list-tree on
/// several postions.

TGListTreeItem* TEveElement::FindListTreeItem(TGListTree* ltree)
{
   for (sLTI_i i = fItems.begin(); i != fItems.end(); ++i)
      if (i->fTree == ltree)
         return i->fItem;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Find list-tree-item of this element with given parent
/// list-tree-item.

TGListTreeItem* TEveElement::FindListTreeItem(TGListTree* ltree,
                                              TGListTreeItem* parent_lti)
{
   for (sLTI_i i = fItems.begin(); i != fItems.end(); ++i)
      if (i->fTree == ltree && i->fItem->GetParent() == parent_lti)
         return i->fItem;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get a TObject associated with this render-element.
/// Most cases uses double-inheritance from TEveElement and TObject
/// so we just do a dynamic cast here.
/// If some TEveElement descendant implements a different scheme,
/// this virtual method should be overriden accordingly.

TObject* TEveElement::GetObject(const TEveException& eh) const
{
   const TObject* obj = dynamic_cast<const TObject*>(this);
   if (obj == 0)
      throw(eh + "not a TObject.");
   return const_cast<TObject*>(obj);
}

////////////////////////////////////////////////////////////////////////////////
/// Show GUI editor for this object.
/// This is forwarded to TEveManager::EditElement().

void TEveElement::SpawnEditor()
{
   gEve->EditElement(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Export render-element to CINT with variable name var_name.

void TEveElement::ExportToCINT(char* var_name)
{
   const char* cname = IsA()->GetName();
   gROOT->ProcessLine(TString::Format("%s* %s = (%s*)0x%lx;", cname, var_name, cname, (ULong_t)this));
}

////////////////////////////////////////////////////////////////////////////////
/// Call Dump() on source object.
/// Throws an exception if it is not set.

void TEveElement::DumpSourceObject() const
{
   static const TEveException eh("TEveElement::DumpSourceObject ");

   TObject *so = GetSourceObject();
   if (!so)
      throw eh + "source-object not set.";

   so->Dump();
}

////////////////////////////////////////////////////////////////////////////////
/// Call Print() on source object.
/// Throws an exception if it is not set.

void TEveElement::PrintSourceObject() const
{
   static const TEveException eh("TEveElement::PrintSourceObject ");

   TObject *so = GetSourceObject();
   if (!so)
      throw eh + "source-object not set.";

   so->Print();
}

////////////////////////////////////////////////////////////////////////////////
/// Export source object to CINT with given name for the variable.
/// Throws an exception if it is not set.

void TEveElement::ExportSourceObjectToCINT(char* var_name) const
{
   static const TEveException eh("TEveElement::ExportSourceObjectToCINT ");

   TObject *so = GetSourceObject();
   if (!so)
      throw eh + "source-object not set.";

   const char* cname = so->IsA()->GetName();
   gROOT->ProcessLine(TString::Format("%s* %s = (%s*)0x%lx;", cname, var_name, cname, (ULong_t)so));
}

////////////////////////////////////////////////////////////////////////////////
/// Paint self and/or children into currently active pad.

void TEveElement::PadPaint(Option_t* option)
{
   static const TEveException eh("TEveElement::PadPaint ");

   TObject* obj = 0;
   if (GetRnrSelf() && (obj = GetRenderObject(eh)))
      obj->Paint(option);


   if (GetRnrChildren()) {
      for (List_i i=BeginChildren(); i!=EndChildren(); ++i) {
         (*i)->PadPaint(option);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint object -- a generic implementation for EVE elements.
/// This supports direct rendering using a dedicated GL class.
/// Override TObject::Paint() in sub-classes if different behaviour
/// is required.

void TEveElement::PaintStandard(TObject* id)
{
   static const TEveException eh("TEveElement::PaintStandard ");

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

////////////////////////////////////////////////////////////////////////////////
/// Set render state of this element, i.e. if it will be published
/// on next scene update pass.
/// Returns true if the state has changed.

Bool_t TEveElement::SetRnrSelf(Bool_t rnr)
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

Bool_t TEveElement::SetRnrChildren(Bool_t rnr)
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

Bool_t TEveElement::SetRnrSelfChildren(Bool_t rnr_self, Bool_t rnr_children)
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

Bool_t TEveElement::SetRnrState(Bool_t rnr)
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
/// Maybe this should be optional on gEve/element level.

void TEveElement::PropagateRnrStateToProjecteds()
{
   TEveProjectable *pable = dynamic_cast<TEveProjectable*>(this);
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

void TEveElement::SetMainColor(Color_t color)
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

void TEveElement::SetMainColorPixel(Pixel_t pixel)
{
   SetMainColor(TColor::GetColor(pixel));
}

////////////////////////////////////////////////////////////////////////////////
/// Convert RGB values to Color_t and call SetMainColor.

void TEveElement::SetMainColorRGB(UChar_t r, UChar_t g, UChar_t b)
{
   SetMainColor(TColor::GetColor(r, g, b));
}

////////////////////////////////////////////////////////////////////////////////
/// Convert RGB values to Color_t and call SetMainColor.

void TEveElement::SetMainColorRGB(Float_t r, Float_t g, Float_t b)
{
   SetMainColor(TColor::GetColor(r, g, b));
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate color to projected elements.

void TEveElement::PropagateMainColorToProjecteds(Color_t color, Color_t old_color)
{
   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->PropagateMainColor(color, old_color);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set main-transparency.
/// Transparency is clamped to [0, 100].

void TEveElement::SetMainTransparency(Char_t t)
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

void TEveElement::SetMainAlpha(Float_t alpha)
{
   if (alpha < 0) alpha = 0;
   if (alpha > 1) alpha = 1;
   SetMainTransparency((Char_t) (100.0f*(1.0f - alpha)));
}

////////////////////////////////////////////////////////////////////////////////
/// Propagate transparency to projected elements.

void TEveElement::PropagateMainTransparencyToProjecteds(Char_t t, Char_t old_t)
{
   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->PropagateMainTransparency(t, old_t);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return pointer to main transformation. If 'create' flag is set (default)
/// it is created if not yet existing.

TEveTrans* TEveElement::PtrMainTrans(Bool_t create)
{
   if (!fMainTrans && create)
      InitMainTrans();

   return fMainTrans;
}

////////////////////////////////////////////////////////////////////////////////
/// Return reference to main transformation. It is created if not yet
/// existing.

TEveTrans& TEveElement::RefMainTrans()
{
   if (!fMainTrans)
      InitMainTrans();

   return *fMainTrans;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the main transformation to identity matrix.
/// If can_edit is true (default), the user will be able to edit the
/// transformation parameters via TEveElementEditor.

void TEveElement::InitMainTrans(Bool_t can_edit)
{
   if (fMainTrans)
      fMainTrans->UnitTrans();
   else
      fMainTrans = new TEveTrans;
   fCanEditMainTrans = can_edit;
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy the main transformation matrix, it will always be taken
/// as identity. Editing of transformation parameters is disabled.

void TEveElement::DestroyMainTrans()
{
   delete fMainTrans;
   fMainTrans = 0;
   fCanEditMainTrans = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set transformation matrix from column-major array.

void TEveElement::SetTransMatrix(Double_t* carr)
{
   RefMainTrans().SetFrom(carr);
}

////////////////////////////////////////////////////////////////////////////////
/// Set transformation matrix from TGeo's matrix.

void TEveElement::SetTransMatrix(const TGeoMatrix& mat)
{
   RefMainTrans().SetFrom(mat);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if el can be added to this element.
///
/// In the base-class version we only make sure the new child is not
/// equal to this.

Bool_t TEveElement::AcceptElement(TEveElement* el)
{
   return el != this;
}

////////////////////////////////////////////////////////////////////////////////
/// Add el to the list of children.

void TEveElement::AddElement(TEveElement* el)
{
   static const TEveException eh("TEveElement::AddElement ");

   if ( ! AcceptElement(el))
      throw(eh + Form("parent '%s' rejects '%s'.",
                      GetElementName(), el->GetElementName()));

   el->AddParent(this);
   fChildren.push_back(el); ++fNumChildren;
   el->AddIntoListTrees(this);
   ElementChanged();
}

////////////////////////////////////////////////////////////////////////////////
/// Remove el from the list of children.

void TEveElement::RemoveElement(TEveElement* el)
{
   el->RemoveFromListTrees(this);
   RemoveElementLocal(el);
   el->RemoveParent(this);
   fChildren.remove(el); --fNumChildren;
   ElementChanged();
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

void TEveElement::RemoveElementLocal(TEveElement* /*el*/)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all elements. This assumes removing of all elements can
/// be done more efficiently then looping over them and removing one
/// by one. This protected function performs the removal on the
/// level of TEveElement.

void TEveElement::RemoveElementsInternal()
{
   for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
   {
      DestroyListSubTree(i->fTree, i->fItem);
   }
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

void TEveElement::RemoveElements()
{
   if (HasChildren())
   {
      RemoveElementsInternal();
      ElementChanged();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Perform additional local removal of all elements.
/// See comment to RemoveElementlocal(TEveElement*).

void TEveElement::RemoveElementsLocal()
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

void TEveElement::ProjectChild(TEveElement* el, Bool_t same_depth)
{
   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(this);
   if (pable && HasChild(el))
   {
      for (TEveProjectable::ProjList_i i = pable->BeginProjecteds(); i != pable->EndProjecteds(); ++i)
      {
         TEveProjectionManager *pmgr = (*i)->GetManager();
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

void TEveElement::ProjectAllChildren(Bool_t same_depth)
{
   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(this);
   if (pable)
   {
      for (TEveProjectable::ProjList_i i = pable->BeginProjecteds(); i != pable->EndProjecteds(); ++i)
      {
         TEveProjectionManager *pmgr = (*i)->GetManager();
         Float_t cd = pmgr->GetCurrentDepth();
         if (same_depth) pmgr->SetCurrentDepth((*i)->GetDepth());

         pmgr->SubImportChildren(this, (*i)->GetProjectedAsElement());

         if (same_depth) pmgr->SetCurrentDepth(cd);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Check if element el is a child of this element.

Bool_t TEveElement::HasChild(TEveElement* el)
{
   return (std::find(fChildren.begin(), fChildren.end(), el) != fChildren.end());
}

////////////////////////////////////////////////////////////////////////////////
/// Find the first child with given name.  If cls is specified (non
/// 0), it is also checked.
///
/// Returns 0 if not found.

TEveElement* TEveElement::FindChild(const TString&  name, const TClass* cls)
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

TEveElement* TEveElement::FindChild(TPRegexp& regexp, const TClass* cls)
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

Int_t TEveElement::FindChildren(List_t& matches,
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

Int_t TEveElement::FindChildren(List_t& matches,
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

TEveElement* TEveElement::FirstChild() const
{
   return HasChildren() ? fChildren.front() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the last child element or 0 if the list is empty.

TEveElement* TEveElement::LastChild () const
{
   return HasChildren() ? fChildren.back() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Enable rendering of children and their list contents.
/// Arguments control how to set self/child rendering.

void TEveElement::EnableListElements(Bool_t rnr_self,  Bool_t rnr_children)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->SetRnrSelf(rnr_self);
      (*i)->SetRnrChildren(rnr_children);
   }

   ElementChanged(kTRUE, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Disable rendering of children and their list contents.
/// Arguments control how to set self/child rendering.
///
/// Same as above function, but default arguments are different. This
/// is convenient for calls via context menu.

void TEveElement::DisableListElements(Bool_t rnr_self,  Bool_t rnr_children)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->SetRnrSelf(rnr_self);
      (*i)->SetRnrChildren(rnr_children);
   }

   ElementChanged(kTRUE, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Protected member function called from TEveElement::Annihilate().

void TEveElement::AnnihilateRecursively()
{
   static const TEveException eh("TEveElement::AnnihilateRecursively ");

   // projected  were already destroyed in TEveElement::Anihilate(), now only clear its list
   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->ClearProjectedList();
   }

   // same as TEveElements::RemoveElementsInternal(), except parents are ignored
   for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
   {
      DestroyListSubTree(i->fTree, i->fItem);
   }
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

void TEveElement::Annihilate()
{
   static const TEveException eh("TEveElement::Annihilate ");

   if (fParents.size() > 1)
   {
      Warning(eh, "More than one parent for '%s': %d. Refusing to delete.",
              GetElementName(), (Int_t) fParents.size());
      return;
   }

   fDestructing = kAnnihilate;

   // recursive annihilation of projecteds
   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(this);
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

   gEve->Redraw3D();
}

////////////////////////////////////////////////////////////////////////////////
/// Annihilate elements.

void TEveElement::AnnihilateElements()
{
   while (!fChildren.empty())
   {
      TEveElement* c = fChildren.front();
      c->Annihilate();
   }

   fNumChildren = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy this element. Throws an exception if deny-destroy is in force.
/// This method should be called instead of a destructor.
/// Note that an exception will be thrown if the element has been
/// protected against destruction with IncDenyDestroy().

void TEveElement::Destroy()
{
   static const TEveException eh("TEveElement::Destroy ");

   if (fDenyDestroy > 0)
      throw eh + TString::Format("element '%s' (%s*) 0x%lx is protected against destruction.",
                                 GetElementName(), IsA()->GetName(), (ULong_t)this);

   PreDeleteElement();
   delete this;
   gEve->Redraw3D();
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy this element. Prints a warning if deny-destroy is in force.

void TEveElement::DestroyOrWarn()
{
   static const TEveException eh("TEveElement::DestroyOrWarn ");

   try
   {
      Destroy();
   }
   catch (TEveException& exc)
   {
      Warning(eh, "%s", exc.Data());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy all children of this element.

void TEveElement::DestroyElements()
{
   static const TEveException eh("TEveElement::DestroyElements ");

   while (HasChildren())
   {
      TEveElement* c = fChildren.front();
      if (c->fDenyDestroy <= 0)
      {
         try {
            c->Destroy();
         }
         catch (const TEveException &exc) {
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

   gEve->Redraw3D();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns state of flag determining if the element will be
/// destroyed when reference count reaches zero.
/// This is true by default.

Bool_t TEveElement::GetDestroyOnZeroRefCnt() const
{
   return fDestroyOnZeroRefCnt;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets the state of flag determining if the element will be
/// destroyed when reference count reaches zero.
/// This is true by default.

void TEveElement::SetDestroyOnZeroRefCnt(Bool_t d)
{
   fDestroyOnZeroRefCnt = d;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the number of times deny-destroy has been requested on
/// the element.

Int_t TEveElement::GetDenyDestroy() const
{
   return fDenyDestroy;
}

////////////////////////////////////////////////////////////////////////////////
/// Increases the deny-destroy count of the element.
/// Call this if you store an external pointer to the element.

void TEveElement::IncDenyDestroy()
{
   ++fDenyDestroy;
}

////////////////////////////////////////////////////////////////////////////////
/// Decreases the deny-destroy count of the element.
/// Call this after releasing an external pointer to the element.

void TEveElement::DecDenyDestroy()
{
   if (--fDenyDestroy <= 0)
      CheckReferenceCount("TEveElement::DecDenyDestroy ");
}

////////////////////////////////////////////////////////////////////////////////
/// Get number of parents that should be ignored in doing
/// reference-counting.
///
/// For example, this is used when subscribing an element to a
/// visualization-database model object.

Int_t TEveElement::GetParentIgnoreCnt() const
{
   return fParentIgnoreCnt;
}

////////////////////////////////////////////////////////////////////////////////
/// Increase number of parents ignored in reference-counting.

void TEveElement::IncParentIgnoreCnt()
{
   ++fParentIgnoreCnt;
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease number of parents ignored in reference-counting.

void TEveElement::DecParentIgnoreCnt()
{
   if (--fParentIgnoreCnt <= 0)
      CheckReferenceCount("TEveElement::DecParentIgnoreCnt ");
}

////////////////////////////////////////////////////////////////////////////////
/// React to element being pasted or dnd-ed.
/// Return true if redraw is needed.

Bool_t TEveElement::HandleElementPaste(TEveElement* el)
{
   gEve->AddElement(el, this);
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Call this after an element has been changed so that the state
/// can be propagated around the framework.

void TEveElement::ElementChanged(Bool_t update_scenes, Bool_t redraw)
{
   gEve->ElementChanged(this, update_scenes, redraw);
}

////////////////////////////////////////////////////////////////////////////////
/// Set pickable state on the element and all its children.

void TEveElement::SetPickableRecursively(Bool_t p)
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
/// TEveSelection.

TEveElement* TEveElement::ForwardSelection()
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns element to be displayed in GUI editor on click.
/// If value is zero the displayed object will follow rules in
/// TEveSelection.

TEveElement* TEveElement::ForwardEdit()
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set element's selection state. Stamp appropriately.

void TEveElement::SelectElement(Bool_t state)
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

void TEveElement::IncImpliedSelected()
{
   if (fImpliedSelected++ == 0)
      StampColorSelection();
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease element's implied-selection count. Stamp appropriately.

void TEveElement::DecImpliedSelected()
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

void TEveElement::UnSelected()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set element's highlight state. Stamp appropriately.

void TEveElement::HighlightElement(Bool_t state)
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

void TEveElement::IncImpliedHighlighted()
{
   if (fImpliedHighlighted++ == 0)
      StampColorSelection();
}

////////////////////////////////////////////////////////////////////////////////
/// Decrease element's implied-highlight count. Stamp appropriately.

void TEveElement::DecImpliedHighlighted()
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

void TEveElement::UnHighlighted()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Populate set impSelSet with derived / dependant elements.
///
/// If this is a TEveProjectable, the projected replicas are added
/// to the set. Thus it does not have to be reimplemented for each
/// sub-class of TEveProjected.
///
/// Note that this also takes care of projections of TEveCompound
/// class, which is also a projectable.

void TEveElement::FillImpliedSelectedSet(Set_t& impSelSet)
{
   TEveProjectable* p = dynamic_cast<TEveProjectable*>(this);
   if (p)
   {
      p->AddProjectedsToSet(impSelSet);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get selection level, needed for rendering selection and
/// highlight feedback.
/// This should go to TAtt3D.

UChar_t TEveElement::GetSelectedLevel() const
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

void TEveElement::RecheckImpliedSelections()
{
   if (fSelected || fImpliedSelected)
      gEve->GetSelection()->RecheckImpliedSetForElement(this);

   if (fHighlighted || fImpliedHighlighted)
      gEve->GetHighlight()->RecheckImpliedSetForElement(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Add (bitwise or) given stamps to fChangeBits.
/// Register this element to gEve as stamped.
/// This method is virtual so that sub-classes can add additional
/// actions. The base-class method should still be called (or replicated).

void TEveElement::AddStamp(UChar_t bits)
{
   fChangeBits |= bits;
   if (fDestructing == kNone) gEve->ElementStamped(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns pointer to first listtreeicon

const TGPicture* TEveElement::GetListTreeIcon(Bool_t open)
{
   // Need better solution for icon-loading/ids !!!!
   return fgListTreeIcons[open ? 7 : 0];
}

////////////////////////////////////////////////////////////////////////////////
/// Returns list-tree-item check-box picture appropriate for given
/// rendering state.

const TGPicture* TEveElement::GetListTreeCheckBoxIcon()
{
   Int_t idx = 0;
   if (fRnrSelf)      idx = 2;
   if (fRnrChildren ) idx++;

   return fgRnrIcons[idx];
}

////////////////////////////////////////////////////////////////////////////////
/// Convert Bool_t to string - kTRUE or kFALSE.
/// Needed in WriteVizParams().

const char* TEveElement::ToString(Bool_t b)
{
   return b ? "kTRUE" : "kFALSE";
}


/** \class  TEveElementList
\ingroup TEve
A list of TEveElements.

Class of acceptable children can be limited by setting the
fChildClass member.

!!! should have two ctors (like in TEveElement), one with Color_t&
and set fDoColor automatically, based on which ctor is called.
*/

ClassImp(TEveElementList);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveElementList::TEveElementList(const char* n, const char* t, Bool_t doColor, Bool_t doTransparency) :
   TEveElement(),
   TNamed(n, t),
   TEveProjectable(),
   fColor(0),
   fChildClass(0)
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

TEveElementList::TEveElementList(const TEveElementList& e) :
   TEveElement (e),
   TNamed      (e),
   TEveProjectable(),
   fColor      (e.fColor),
   fChildClass (e.fChildClass)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Clone the element via copy constructor.
/// Virtual from TEveElement.

TEveElementList* TEveElementList::CloneElement() const
{
   return new TEveElementList(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Check if TEveElement el is inherited from fChildClass.
/// Virtual from TEveElement.

Bool_t TEveElementList::AcceptElement(TEveElement* el)
{
   if (fChildClass && ! el->IsA()->InheritsFrom(fChildClass))
      return kFALSE;
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveProjectable, returns TEveCompoundProjected class.

TClass* TEveElementList::ProjectedClass(const TEveProjection*) const
{
   return TEveElementListProjected::Class();
}

/** \class TEveElementListProjected
\ingroup TEve
A projected element list -- required for proper propagation
of render state to projected views.
*/

ClassImp(TEveElementListProjected);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveElementListProjected::TEveElementListProjected() :
   TEveElementList("TEveElementListProjected")
{
}

////////////////////////////////////////////////////////////////////////////////
/// This is abstract method from base-class TEveProjected.
/// No implementation.

void TEveElementListProjected::UpdateProjection()
{
}
