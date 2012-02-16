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

//==============================================================================
//==============================================================================
// TEveElement::TEveListTreeInfo
//==============================================================================

//______________________________________________________________________________
//
// Structure holding information about TGListTree and TGListTreeItem
// that represents given TEveElement. This needed because each element
// can appear in several list-trees as well as several times in the
// same list-tree.

ClassImp(TEveElement::TEveListTreeInfo);


//==============================================================================
//==============================================================================
// TEveElement
//==============================================================================

//______________________________________________________________________________
//
// Base class for TEveUtil visualization elements, providing hierarchy
// management, rendering control and list-tree item management.

ClassImp(TEveElement);

//______________________________________________________________________________
const TGPicture* TEveElement::fgRnrIcons[4]      = { 0 };
const TGPicture* TEveElement::fgListTreeIcons[9] = { 0 };

//______________________________________________________________________________
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
   // Default contructor.
}

//______________________________________________________________________________
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
   // Constructor.
}

//______________________________________________________________________________
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
   // Copy constructor. Does shallow copy.
   // For deep-cloning and children-cloning, see:
   //   TEveElement* CloneElementRecurse(Int_t level)
   //   void         CloneChildrenRecurse(TEveElement* dest, Int_t level)
   //
   // 'TRef fSource' is copied but 'void* UserData' is NOT.
   // If the element is projectable, its projections are NOT copied.
   //
   // Not implemented for most sub-classes, let us know.
   // Note that sub-classes of TEveProjected are NOT and will NOT be copyable.

   SetVizModel(e.fVizModel);
   if (e.fMainColorPtr)
      fMainColorPtr = (Color_t*)((const char*) this + ((const char*) e.fMainColorPtr - (const char*) &e));
   if (e.fMainTrans)
      fMainTrans = new TEveTrans(*e.fMainTrans);
}

//______________________________________________________________________________
TEveElement::~TEveElement()
{
   // Destructor.
  
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

//______________________________________________________________________________
void TEveElement::PreDeleteElement()
{
   // Called before the element is deleted, thus offering the last chance
   // to detach from acquired resources and from the framework itself.
   // Here the request is just passed to TEveManager.
   // If you override it, make sure to call base-class version.

   gEve->PreDeleteElement(this);
}

//______________________________________________________________________________
TEveElement* TEveElement::CloneElement() const
{
   // Clone the element via copy constructor.
   // Should be implemented for all classes that require cloning support.

   return new TEveElement(*this);
}

//______________________________________________________________________________
TEveElement* TEveElement::CloneElementRecurse(Int_t level) const
{
   // Clone elements and recurse 'level' deep over children.
   // If level ==  0, only the element itself is cloned (default).
   // If level == -1, all the hierarchy is cloned.

   TEveElement* el = CloneElement();
   if (level--)
   {
      CloneChildrenRecurse(el, level);
   }
   return el;
}

//______________________________________________________________________________
void TEveElement::CloneChildrenRecurse(TEveElement* dest, Int_t level) const
{
   // Clone children and attach them to the dest element.
   // If level ==  0, only the direct descendants are cloned (default).
   // If level == -1, all the hierarchy is cloned.

   for (List_ci i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      dest->AddElement((*i)->CloneElementRecurse(level));
   }
}


//==============================================================================

//______________________________________________________________________________
const char* TEveElement::GetElementName() const
{
   // Virtual function for retrieveing name of the element.
   // Here we attempt to cast the assigned object into TNamed and call
   // GetName() there.

   static const TEveException eh("TEveElement::GetElementName ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   return named ? named->GetName() : "<no-name>";
}

//______________________________________________________________________________
const char*  TEveElement::GetElementTitle() const
{
   // Virtual function for retrieveing title of the render-element.
   // Here we attempt to cast the assigned object into TNamed and call
   // GetTitle() there.

   static const TEveException eh("TEveElement::GetElementTitle ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   return named ? named->GetTitle() : "<no-title>";
}

//______________________________________________________________________________
void TEveElement::SetElementName(const char* name)
{
   // Virtual function for setting of name of an element.
   // Here we attempt to cast the assigned object into TNamed and call
   // SetName() there.
   // If you override this call NameTitleChanged() from there.

   static const TEveException eh("TEveElement::SetElementName ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   if (named) {
      named->SetName(name);
      NameTitleChanged();
   }
}

//______________________________________________________________________________
void TEveElement::SetElementTitle(const char* title)
{
   // Virtual function for setting of title of an element.
   // Here we attempt to cast the assigned object into TNamed and call
   // SetTitle() there.
   // If you override this call NameTitleChanged() from there.

   static const TEveException eh("TEveElement::SetElementTitle ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   if (named) {
      named->SetTitle(title);
      NameTitleChanged();
   }
}

//______________________________________________________________________________
void TEveElement::SetElementNameTitle(const char* name, const char* title)
{
   // Virtual function for setting of name and title of render element.
   // Here we attempt to cast the assigned object into TNamed and call
   // SetNameTitle() there.
   // If you override this call NameTitleChanged() from there.

   static const TEveException eh("TEveElement::SetElementNameTitle ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   if (named) {
      named->SetNameTitle(name, title);
      NameTitleChanged();
   }
}

//______________________________________________________________________________
void TEveElement::NameTitleChanged()
{
   // Virtual function called when a name or title of the element has
   // been changed.
   // If you override this, call also the version of your direct base-class.

   // Nothing to do - list-tree-items take this info directly.
}

//******************************************************************************

//______________________________________________________________________________
void TEveElement::SetVizModel(TEveElement* model)
{
   // Set visualization-parameter model element.
   // Calling of this function from outside of EVE should in principle
   // be avoided as it can lead to dis-synchronization of viz-tag and
   // viz-model.

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

//______________________________________________________________________________
Bool_t TEveElement::FindVizModel()
{
   // Find model element in VizDB that corresponds to previously
   // assigned fVizTag and set fVizModel accordingly.
   // If the tag is not found in VizDB, the old model-element is kept
   // and false is returned.

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

//______________________________________________________________________________
Bool_t TEveElement::ApplyVizTag(const TString& tag, const TString& fallback_tag)
{
   // Set the VizTag, find model-element from the VizDB and copy
   // visualization-parameters from it. If the model is not found and
   // fallback_tag is non-null, its search is attempted as well.
   // For example: ApplyVizTag("TPC Clusters", "Clusters");
   //
   // If the model-element can not be found a warning is printed and
   // false is returned.

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

//______________________________________________________________________________
void TEveElement::PropagateVizParamsToProjecteds()
{
   // Propagate visualization parameters to dependent elements.
   //
   // MainColor is propagated independently in SetMainColor().
   // In this case, as fMainColor is a pointer to Color_t, it should
   // be set in TProperClass::CopyVizParams().
   //
   // Render state is not propagated. Maybe it should be, at least optionally.

   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->PropagateVizParams();
   }
}

//______________________________________________________________________________
void TEveElement::PropagateVizParamsToElements(TEveElement* el)
{
   // Propagate visualization parameters from element el (defaulting
   // to this) to all elements (children).
   //
   // The primary use of this is for model-elements from
   // visualization-parameter database.

   if (el == 0)
      el = this;

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->CopyVizParams(el);
   }
}

//______________________________________________________________________________
void TEveElement::CopyVizParams(const TEveElement* el)
{
   // Copy visualization parameters from element el.
   // This method needs to be overriden by any class that introduces
   // new parameters.
   // Color is copied in sub-classes which define it.
   // See, for example, TEvePointSet::CopyVizParams(),
   // TEveLine::CopyVizParams() and TEveTrack::CopyVizParams().

   fCanEditMainColor        = el->fCanEditMainColor;
   fCanEditMainTransparency = el->fCanEditMainTransparency;
   fMainTransparency        = el->fMainTransparency;

   AddStamp(kCBColorSelection | kCBObjProps);
}

//______________________________________________________________________________
void TEveElement::CopyVizParamsFromDB()
{
   // Copy visualization parameters from the model-element fVizModel.
   // A warning is printed if the model-element fVizModel is not set.

   if (fVizModel)
   {
      CopyVizParams(fVizModel);
   }
   else
   {
      Warning("TEveElement::CopyVizParamsFromDB", "VizModel has not been set.");
   }
}

//______________________________________________________________________________
void TEveElement::SaveVizParams(ostream& out, const TString& tag, const TString& var)
{
   // Save visualization parameters for this element with given tag.
   //
   // This function creates the instantiation code, calls virtual
   // WriteVizParams() and, at the end, writes out the code for
   // registration of the model into the VizDB.

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

//______________________________________________________________________________
void TEveElement::WriteVizParams(ostream& out, const TString& var)
{
   // Write-out visual parameters for this object.
   // This is a virtual function and all sub-classes are required to
   // first call the base-element version.
   // The name of the element pointer is 'x%03d', due to cint limitations.
   // Three spaces should be used for indentation, same as in
   // SavePrimitive() methods.

   TString t = "   " + var + "->";

   out << t << "SetElementName(\""  << GetElementName()  << "\");\n";
   out << t << "SetElementTitle(\"" << GetElementTitle() << "\");\n";
   out << t << "SetEditMainColor("  << fCanEditMainColor << ");\n";
   out << t << "SetEditMainTransparency(" << fCanEditMainTransparency << ");\n";
   out << t << "SetMainTransparency("     << fMainTransparency << ");\n";
}

//______________________________________________________________________________
void TEveElement::VizDB_Apply(const char* tag)
{
   // Set visual parameters for this object for given tag.

   if (ApplyVizTag(tag))
   {
      PropagateVizParamsToProjecteds();
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void TEveElement::VizDB_Reapply()
{
   // Reset visual parameters for this object from VizDB.
   // The model object must be already set.

   if (fVizModel)
   {
      CopyVizParamsFromDB();
      PropagateVizParamsToProjecteds();
      gEve->Redraw3D();
   }
}

//______________________________________________________________________________
void TEveElement::VizDB_UpdateModel(Bool_t update)
{
   // Copy visual parameters from this element to viz-db model.
   // If update is set, all clients of the model will be updated to
   // the new value.
   // A warning is printed if the model-element fVizModel is not set.

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

//______________________________________________________________________________
void TEveElement::VizDB_Insert(const char* tag, Bool_t replace, Bool_t update)
{
   // Create a replica of element and insert it into VizDB with given tag.
   // If replace is true an existing element with the same tag will be replaced.
   // If update is true, existing client of tag will be updated.

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

//******************************************************************************

//______________________________________________________________________________
TEveElement* TEveElement::GetMaster()
{
   // Returns the master element - that is:
   // - master of projectable, if this is a projected;
   // - master of compound, if fCompound is set;
   // - master of first compound parent, if kSCBTakeAnyParentAsMaster bit is set;
   // If non of the above is true, *this* is returned.

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

//******************************************************************************

//______________________________________________________________________________
void TEveElement::AddParent(TEveElement* re)
{
   // Add re into the list parents.
   // Adding parent is subordinate to adding an element.
   // This is an internal function.

   fParents.push_back(re);
}

//______________________________________________________________________________
void TEveElement::RemoveParent(TEveElement* re)
{
   // Remove re from the list of parents.
   // Removing parent is subordinate to removing an element.
   // This is an internal function.

   static const TEveException eh("TEveElement::RemoveParent ");

   fParents.remove(re);
   CheckReferenceCount(eh);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElement::CheckReferenceCount(const TEveException& eh)
{
   // Check external references to this and eventually auto-destruct
   // the render-element.

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

//______________________________________________________________________________
void TEveElement::CollectSceneParents(List_t& scenes)
{
   // Collect all parents of class TEveScene. This is needed to
   // automatically detect which scenes need to be updated.
   //
   // Overriden in TEveScene to include itself and return.

   for (List_i p=fParents.begin(); p!=fParents.end(); ++p)
      (*p)->CollectSceneParents(scenes);
}

//______________________________________________________________________________
void TEveElement::CollectSceneParentsFromChildren(List_t&      scenes,
                                                  TEveElement* parent)
{
   // Collect scene-parents from all children. This is needed to
   // automatically detect which scenes need to be updated during/after
   // a full sub-tree update.
   // Argument parent specifies parent in traversed hierarchy for which we can
   // skip the upwards search.

   for (List_i p=fParents.begin(); p!=fParents.end(); ++p)
   {
      if (*p != parent) (*p)->CollectSceneParents(scenes);
   }

   for (List_i c=fChildren.begin(); c!=fChildren.end(); ++c)
   {
      (*c)->CollectSceneParentsFromChildren(scenes, this);
   }
}

/******************************************************************************/
// List-tree stuff
/******************************************************************************/

//______________________________________________________________________________
void TEveElement::ExpandIntoListTree(TGListTree* ltree,
                                     TGListTreeItem* parent)
{
   // Populates parent with elements.
   // parent must be an already existing representation of *this*.
   // Returns number of inserted elements.
   // If parent already has children, it does nothing.
   //
   // Element can be inserted in a list-tree several times, thus we can not
   // search through fItems to get parent here.
   // Anyhow, it is probably known as it must have been selected by the user.

   if (parent->GetFirstChild() != 0)
      return;
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i) {
      (*i)->AddIntoListTree(ltree, parent);
   }
}

//______________________________________________________________________________
void TEveElement::DestroyListSubTree(TGListTree* ltree,
                                     TGListTreeItem* parent)
{
   // Destroy sub-tree under item 'parent' in list-tree 'ltree'.

   TGListTreeItem* i = parent->GetFirstChild();
   while (i != 0)
   {
      TEveElement* re = (TEveElement*) i->GetUserData();
      i = i->GetNextSibling();
      re->RemoveFromListTree(ltree, parent);
   }
}

//______________________________________________________________________________
TGListTreeItem* TEveElement::AddIntoListTree(TGListTree* ltree,
                                             TGListTreeItem* parent_lti)
{
   // Add this element into ltree to an already existing item
   // parent_lti. Children, if any, are added as below the newly created item.
   // Returns the newly created list-tree-item.

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

//______________________________________________________________________________
TGListTreeItem* TEveElement::AddIntoListTree(TGListTree* ltree,
                                             TEveElement* parent)
{
   // Add this render element into ltree to all items belonging to
   // parent. Returns list-tree-item from the first register entry (but
   // we use a set for that so it can be anything).

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

//______________________________________________________________________________
TGListTreeItem* TEveElement::AddIntoListTrees(TEveElement* parent)
{
   // Add this render element into all list-trees and all items
   // belonging to parent. Returns list-tree-item from the first
   // register entry (but we use a set for that so it can be anything).

   TGListTreeItem* lti = 0;
   for (sLTI_ri i = parent->fItems.rbegin(); i != parent->fItems.rend(); ++i)
   {
      lti = AddIntoListTree(i->fTree, i->fItem);
   }
   return lti;
}

//______________________________________________________________________________
Bool_t TEveElement::RemoveFromListTree(TGListTree* ltree,
                                       TGListTreeItem* parent_lti)
{
   // Remove element from list-tree 'ltree' where its parent item is
   // 'parent_lti'.
   // Returns kTRUE if the item was found and removed, kFALSE
   // otherwise.

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

//______________________________________________________________________________
Int_t TEveElement::RemoveFromListTrees(TEveElement* parent)
{
   // Remove element from all list-trees where 'parent' is the
   // user-data of the parent list-tree-item.

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

//______________________________________________________________________________
TEveElement::sLTI_i TEveElement::FindItem(TGListTree* ltree)
{
   // Find any list-tree-item of this element in list-tree 'ltree'.
   // Note that each element can be placed into the same list-tree on
   // several postions.

   for (sLTI_i i = fItems.begin(); i != fItems.end(); ++i)
      if (i->fTree == ltree)
         return i;
   return fItems.end();
}

//______________________________________________________________________________
TEveElement::sLTI_i TEveElement::FindItem(TGListTree* ltree,
                                          TGListTreeItem* parent_lti)
{
   // Find list-tree-item of this element with given parent
   // list-tree-item.

   for (sLTI_i i = fItems.begin(); i != fItems.end(); ++i)
      if (i->fTree == ltree && i->fItem->GetParent() == parent_lti)
         return i;
   return fItems.end();
}

//______________________________________________________________________________
TGListTreeItem* TEveElement::FindListTreeItem(TGListTree* ltree)
{
   // Find any list-tree-item of this element in list-tree 'ltree'.
   // Note that each element can be placed into the same list-tree on
   // several postions.

   for (sLTI_i i = fItems.begin(); i != fItems.end(); ++i)
      if (i->fTree == ltree)
         return i->fItem;
   return 0;
}

//______________________________________________________________________________
TGListTreeItem* TEveElement::FindListTreeItem(TGListTree* ltree,
                                              TGListTreeItem* parent_lti)
{
   // Find list-tree-item of this element with given parent
   // list-tree-item.

   for (sLTI_i i = fItems.begin(); i != fItems.end(); ++i)
      if (i->fTree == ltree && i->fItem->GetParent() == parent_lti)
         return i->fItem;
   return 0;
}

/******************************************************************************/

//______________________________________________________________________________
TObject* TEveElement::GetObject(const TEveException& eh) const
{
   // Get a TObject associated with this render-element.
   // Most cases uses double-inheritance from TEveElement and TObject
   // so we just do a dynamic cast here.
   // If some TEveElement descendant implements a different scheme,
   // this virtual method should be overriden accordingly.

   const TObject* obj = dynamic_cast<const TObject*>(this);
   if (obj == 0)
      throw(eh + "not a TObject.");
   return const_cast<TObject*>(obj);
}

//______________________________________________________________________________
void TEveElement::SpawnEditor()
{
   // Show GUI editor for this object.
   // This is forwarded to TEveManager::EditElement().

   gEve->EditElement(this);
}

//______________________________________________________________________________
void TEveElement::ExportToCINT(char* var_name)
{
   // Export render-element to CINT with variable name var_name.

   const char* cname = IsA()->GetName();
   gROOT->ProcessLine(TString::Format("%s* %s = (%s*)0x%lx;", cname, var_name, cname, (ULong_t)this));
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElement::DumpSourceObject() const
{
   // Call Dump() on source object.
   // Throws an exception if it is not set.

   static const TEveException eh("TEveElement::DumpSourceObject ");

   TObject *so = GetSourceObject();
   if (!so)
      throw eh + "source-object not set.";

   so->Dump();
}

//______________________________________________________________________________
void TEveElement::PrintSourceObject() const
{
   // Call Print() on source object.
   // Throws an exception if it is not set.

   static const TEveException eh("TEveElement::PrintSourceObject ");

   TObject *so = GetSourceObject();
   if (!so)
      throw eh + "source-object not set.";

   so->Print();
}

//______________________________________________________________________________
void TEveElement::ExportSourceObjectToCINT(char* var_name) const
{
   // Export source object to CINT with given name for the variable.
   // Throws an exception if it is not set.

   static const TEveException eh("TEveElement::ExportSourceObjectToCINT ");

   TObject *so = GetSourceObject();
   if (!so)
      throw eh + "source-object not set.";

   const char* cname = so->IsA()->GetName();
   gROOT->ProcessLine(TString::Format("%s* %s = (%s*)0x%lx;", cname, var_name, cname, (ULong_t)so));
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElement::PadPaint(Option_t* option)
{
   // Paint self and/or children into currently active pad.

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

//______________________________________________________________________________
void TEveElement::PaintStandard(TObject* id)
{
   // Paint object -- a generic implementation for EVE elements.
   // This supports direct rendering using a dedicated GL class.
   // Override TObject::Paint() in sub-classes if different behaviour
   // is required.

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

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveElement::SetRnrSelf(Bool_t rnr)
{
   // Set render state of this element, i.e. if it will be published
   // on next scene update pass.
   // Returns true if the state has changed.

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

//______________________________________________________________________________
Bool_t TEveElement::SetRnrChildren(Bool_t rnr)
{
   // Set render state of this element's children, i.e. if they will
   // be published on next scene update pass.
   // Returns true if the state has changed.

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

//______________________________________________________________________________
Bool_t TEveElement::SetRnrSelfChildren(Bool_t rnr_self, Bool_t rnr_children)
{
   // Set state for rendering of this element and its children.
   // Returns true if the state has changed.

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

//______________________________________________________________________________
Bool_t TEveElement::SetRnrState(Bool_t rnr)
{
   // Set render state of this element and of its children to the same
   // value.
   // Returns true if the state has changed.

   if (fRnrSelf != rnr || fRnrChildren != rnr)
   {
      fRnrSelf = fRnrChildren = rnr;
      StampVisibility();
      PropagateRnrStateToProjecteds();
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TEveElement::PropagateRnrStateToProjecteds()
{
   // Propagate render state to the projected replicas of this element.
   // Maybe this should be optional on gEve/element level.

   TEveProjectable *pable = dynamic_cast<TEveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->PropagateRenderState(fRnrSelf, fRnrChildren);
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElement::SetMainColor(Color_t color)
{
   // Set main color of the element.
   //
   //
   // List-tree-items are updated.

   Color_t old_color = GetMainColor();

   if (fMainColorPtr)
   {
      *fMainColorPtr = color;
      StampColorSelection();
   }

   PropagateMainColorToProjecteds(color, old_color);
}

//______________________________________________________________________________
void TEveElement::SetMainColorPixel(Pixel_t pixel)
{
   // Convert pixel to Color_t and call SetMainColor().

   SetMainColor(TColor::GetColor(pixel));
}

//______________________________________________________________________________
void TEveElement::SetMainColorRGB(UChar_t r, UChar_t g, UChar_t b)
{
   // Convert RGB values to Color_t and call SetMainColor.

   SetMainColor(TColor::GetColor(r, g, b));
}

//______________________________________________________________________________
void TEveElement::SetMainColorRGB(Float_t r, Float_t g, Float_t b)
{
   // Convert RGB values to Color_t and call SetMainColor.

   SetMainColor(TColor::GetColor(r, g, b));
}

//______________________________________________________________________________
void TEveElement::PropagateMainColorToProjecteds(Color_t color, Color_t old_color)
{
   // Propagate color to projected elements.

   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->PropagateMainColor(color, old_color);
   }
}

//______________________________________________________________________________
void TEveElement::SetMainTransparency(Char_t t)
{
   // Set main-transparency.
   // Transparency is clamped to [0, 100].

   Char_t old_t = GetMainTransparency();

   if (t > 100) t = 100;
   fMainTransparency = t;
   StampColorSelection();

   PropagateMainTransparencyToProjecteds(t, old_t);
}

//______________________________________________________________________________
void TEveElement::SetMainAlpha(Float_t alpha)
{
   // Set main-transparency via float alpha varable.
   // Value of alpha is clamped t0 [0, 1].

   if (alpha < 0) alpha = 0;
   if (alpha > 1) alpha = 1;
   SetMainTransparency((Char_t) (100.0f*(1.0f - alpha)));
}

//______________________________________________________________________________
void TEveElement::PropagateMainTransparencyToProjecteds(Char_t t, Char_t old_t)
{
   // Propagate transparency to projected elements.

   TEveProjectable* pable = dynamic_cast<TEveProjectable*>(this);
   if (pable && pable->HasProjecteds())
   {
      pable->PropagateMainTransparency(t, old_t);
   }
}


/******************************************************************************/

//______________________________________________________________________________
TEveTrans* TEveElement::PtrMainTrans(Bool_t create)
{
   // Return pointer to main transformation. If 'create' flag is set (default)
   // it is created if not yet existing.

   if (!fMainTrans && create)
      InitMainTrans();

   return fMainTrans;
}

//______________________________________________________________________________
TEveTrans& TEveElement::RefMainTrans()
{
   // Return reference to main transformation. It is created if not yet
   // existing.

   if (!fMainTrans)
      InitMainTrans();

   return *fMainTrans;
}

//______________________________________________________________________________
void TEveElement::InitMainTrans(Bool_t can_edit)
{
   // Initialize the main transformation to identity matrix.
   // If can_edit is true (default), the user will be able to edit the
   // transformation parameters via TEveElementEditor.

   if (fMainTrans)
      fMainTrans->UnitTrans();
   else
      fMainTrans = new TEveTrans;
   fCanEditMainTrans = can_edit;
}

//______________________________________________________________________________
void TEveElement::DestroyMainTrans()
{
   // Destroy the main transformation matrix, it will always be taken
   // as identity. Editing of transformation parameters is disabled.

   delete fMainTrans;
   fMainTrans = 0;
   fCanEditMainTrans = kFALSE;
}

//______________________________________________________________________________
void TEveElement::SetTransMatrix(Double_t* carr)
{
   // Set transformation matrix from colum-major array.

   RefMainTrans().SetFrom(carr);
}

//______________________________________________________________________________
void TEveElement::SetTransMatrix(const TGeoMatrix& mat)
{
   // Set transformation matrix from TGeo's matrix.

   RefMainTrans().SetFrom(mat);
}


/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveElement::AcceptElement(TEveElement* el)
{
   // Check if el can be added to this element.
   //
   // In the base-class version we only make sure the new child is not
   // equal to this.

   return el != this;
}

//______________________________________________________________________________
void TEveElement::AddElement(TEveElement* el)
{
   // Add el to the list of children.

   static const TEveException eh("TEveElement::AddElement ");

   if ( ! AcceptElement(el))
      throw(eh + Form("parent '%s' rejects '%s'.",
                      GetElementName(), el->GetElementName()));

   el->AddParent(this);
   fChildren.push_back(el); ++fNumChildren;
   el->AddIntoListTrees(this);
   ElementChanged();
}

//______________________________________________________________________________
void TEveElement::RemoveElement(TEveElement* el)
{
   // Remove el from the list of children.

   el->RemoveFromListTrees(this);
   RemoveElementLocal(el);
   el->RemoveParent(this);
   fChildren.remove(el); --fNumChildren;
   ElementChanged();
}

//______________________________________________________________________________
void TEveElement::RemoveElementLocal(TEveElement* /*el*/)
{
   // Perform additional local removal of el.
   // Called from RemoveElement() which does whole untangling.
   // Put into special function as framework-related handling of
   // element removal should really be common to all classes and
   // clearing of local structures happens in between removal
   // of list-tree-items and final removal.
   // If you override this, you should also override
   // RemoveElementsLocal().
}

//______________________________________________________________________________
void TEveElement::RemoveElementsInternal()
{
   // Remove all elements. This assumes removing of all elements can
   // be done more efficiently then looping over them and removing one
   // by one. This protected function performs the removal on the
   // level of TEveElement.

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

//______________________________________________________________________________
void TEveElement::RemoveElements()
{
   // Remove all elements. This assumes removing of all elements can
   // be done more efficiently then looping over them and removing
   // them one by one.

   if (HasChildren())
   {
      RemoveElementsInternal();
      ElementChanged();
   }
}

//______________________________________________________________________________
void TEveElement::RemoveElementsLocal()
{
   // Perform additional local removal of all elements.
   // See comment to RemoveElementlocal(TEveElement*).
}

//==============================================================================

//______________________________________________________________________________
void TEveElement::ProjectChild(TEveElement* el, Bool_t same_depth)
{
   // If this is a projectable, loop over all projected replicas and
   // add the projected image of child 'el' there. This is supposed to
   // be called after you add a child to a projectable after it has
   // already been projected.
   // You might also want to call RecheckImpliedSelections() on this
   // element or 'el'.
   //
   // If 'same_depth' flag is true, the same depth as for parent object
   // is used in every projection. Otherwise current depth of each
   // relevant projection-manager is used.

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

//______________________________________________________________________________
void TEveElement::ProjectAllChildren(Bool_t same_depth)
{
   // If this is a projectable, loop over all projected replicas and
   // add the projected image of all children there. This is supposed
   // to be called after you destroy all children and then add new
   // ones after this element has already been projected.
   // You might also want to call RecheckImpliedSelections() on this
   // element.
   //
   // If 'same_depth' flag is true, the same depth as for the
   // projected element is used in every projection. Otherwise current
   // depth of each relevant projection-manager is used.

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

//==============================================================================

//______________________________________________________________________________
Bool_t TEveElement::HasChild(TEveElement* el)
{
   // Check if element el is a child of this element.

   return (std::find(fChildren.begin(), fChildren.end(), el) != fChildren.end());
}

//______________________________________________________________________________
TEveElement* TEveElement::FindChild(const TString&  name, const TClass* cls)
{
   // Find the first child with given name.  If cls is specified (non
   // 0), it is also checked.
   //
   // Returns 0 if not found.

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

//______________________________________________________________________________
TEveElement* TEveElement::FindChild(TPRegexp& regexp, const TClass* cls)
{
   // Find the first child whose name matches regexp. If cls is
   // specified (non 0), it is also checked.
   //
   // Returns 0 if not found.

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

//______________________________________________________________________________
Int_t TEveElement::FindChildren(List_t& matches,
                                const TString& name, const TClass* cls)
{
   // Find all children with given name and append them to matches
   // list. If class is specified (non 0), it is also checked.
   //
   // Returns number of elements added to the list.

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

//______________________________________________________________________________
Int_t TEveElement::FindChildren(List_t& matches,
                                TPRegexp& regexp, const TClass* cls)
{
   // Find all children whose name matches regexp and append them to
   // matches list.
   //
   // Returns number of elements added to the list.

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

//______________________________________________________________________________
TEveElement* TEveElement::FirstChild() const
{
   // Returns the first child element or 0 if the list is empty.

   return HasChildren() ? fChildren.front() : 0;
}

//______________________________________________________________________________
TEveElement* TEveElement::LastChild () const
{
   // Returns the last child element or 0 if the list is empty.

   return HasChildren() ? fChildren.back() : 0;
}


//==============================================================================

//______________________________________________________________________________
void TEveElement::EnableListElements(Bool_t rnr_self,  Bool_t rnr_children)
{
   // Enable rendering of children and their list contents.
   // Arguments control how to set self/child rendering.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->SetRnrSelf(rnr_self);
      (*i)->SetRnrChildren(rnr_children);
   }

   ElementChanged(kTRUE, kTRUE);
}

//______________________________________________________________________________
void TEveElement::DisableListElements(Bool_t rnr_self,  Bool_t rnr_children)
{
   // Disable rendering of children and their list contents.
   // Arguments control how to set self/child rendering.
   //
   // Same as above function, but default arguments are different. This
   // is convenient for calls via context menu.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->SetRnrSelf(rnr_self);
      (*i)->SetRnrChildren(rnr_children);
   }

   ElementChanged(kTRUE, kTRUE);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElement::AnnihilateRecursively()
{
   // Protected member function called from TEveElement::Annihilate().

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

//______________________________________________________________________________
void TEveElement::Annihilate()
{
   // Optimized destruction without check of reference-count.
   // Parents are not notified about child destruction. 
   // The method should only be used when an element does not have
   // more than one parent -- otherwise an exception is thrown.

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

//______________________________________________________________________________
void TEveElement::AnnihilateElements()
{  
   // Annihilate elements.
   
   while (!fChildren.empty())
   {
      TEveElement* c = fChildren.front();
      c->Annihilate();
   }

   fNumChildren = 0;
}

//______________________________________________________________________________
void TEveElement::Destroy()
{
   // Destroy this element. Throws an exception if deny-destroy is in force.

   static const TEveException eh("TEveElement::Destroy ");

   if (fDenyDestroy > 0)
      throw eh + TString::Format("element '%s' (%s*) 0x%lx is protected against destruction.",
                                 GetElementName(), IsA()->GetName(), (ULong_t)this);

   PreDeleteElement();
   delete this;
   gEve->Redraw3D();
}

//______________________________________________________________________________
void TEveElement::DestroyOrWarn()
{
   // Destroy this element. Prints a warning if deny-destroy is in force.

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

//______________________________________________________________________________
void TEveElement::DestroyElements()
{
   // Destroy all children of this element.

   static const TEveException eh("TEveElement::DestroyElements ");

   while (HasChildren())
   {
      TEveElement* c = fChildren.front();
      if (c->fDenyDestroy <= 0)
      {
         try {
            c->Destroy();
         }
         catch (TEveException exc) {
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

//______________________________________________________________________________
Bool_t TEveElement::GetDestroyOnZeroRefCnt() const
{
   // Returns state of flag determining if the element will be
   // destroyed when reference count reaches zero.
   // This is true by default.

   return fDestroyOnZeroRefCnt;
}

//______________________________________________________________________________
void TEveElement::SetDestroyOnZeroRefCnt(Bool_t d)
{
   // Sets the state of flag determining if the element will be
   // destroyed when reference count reaches zero.
   // This is true by default.

   fDestroyOnZeroRefCnt = d;
}

//______________________________________________________________________________
Int_t TEveElement::GetDenyDestroy() const
{
   // Returns the number of times deny-destroy has been requested on
   // the element.

   return fDenyDestroy;
}

//______________________________________________________________________________
void TEveElement::IncDenyDestroy()
{
   // Increases the deny-destroy count of the element.
   // Call this if you store an external pointer to the element.

   ++fDenyDestroy;
}

//______________________________________________________________________________
void TEveElement::DecDenyDestroy()
{
   // Decreases the deny-destroy count of the element.
   // Call this after releasing an external pointer to the element.

   if (--fDenyDestroy <= 0)
      CheckReferenceCount("TEveElement::DecDenyDestroy ");
}

//______________________________________________________________________________
Int_t TEveElement::GetParentIgnoreCnt() const
{
   // Get number of parents that should be ignored in doing
   // reference-counting.
   //
   // For example, this is used when subscribing an element to a
   // visualization-database model object.

   return fParentIgnoreCnt;
}

//______________________________________________________________________________
void TEveElement::IncParentIgnoreCnt()
{
   // Increase number of parents ignored in reference-counting.

   ++fParentIgnoreCnt;
}

//______________________________________________________________________________
void TEveElement::DecParentIgnoreCnt()
{
   // Decrease number of parents ignored in reference-counting.

   if (--fParentIgnoreCnt <= 0)
      CheckReferenceCount("TEveElement::DecParentIgnoreCnt ");
}


//==============================================================================

//______________________________________________________________________________
Bool_t TEveElement::HandleElementPaste(TEveElement* el)
{
   // React to element being pasted or dnd-ed.
   // Return true if redraw is needed.

   gEve->AddElement(el, this);
   return kTRUE;
}

//______________________________________________________________________________
void TEveElement::ElementChanged(Bool_t update_scenes, Bool_t redraw)
{
   // Call this after an element has been changed so that the state
   // can be propagated around the framework.

   gEve->ElementChanged(this, update_scenes, redraw);
}

/******************************************************************************/
// Select/hilite
/******************************************************************************/

//______________________________________________________________________________
void TEveElement::SetPickableRecursively(Bool_t p)
{
   // Set pickable state on the element and all its children.

   fPickable = p;
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->SetPickableRecursively(p);
   }
}

//______________________________________________________________________________
TEveElement* TEveElement::ForwardSelection()
{
   // Returns element to be selected on click.
   // If value is zero the selected object will follow rules in
   // TEveSelection.

   return 0;
}

//______________________________________________________________________________
TEveElement* TEveElement::ForwardEdit()
{
   // Returns element to be displayed in GUI editor on click.
   // If value is zero the displayed object will follow rules in
   // TEveSelection.

   return 0;
}

//______________________________________________________________________________
void TEveElement::SelectElement(Bool_t state)
{
   // Set element's selection state. Stamp appropriately.

   if (fSelected != state) {
      fSelected = state;
      if (!fSelected && fImpliedSelected == 0)
         UnSelected();
      fParentIgnoreCnt += (fSelected) ? 1 : -1;
      StampColorSelection();
   }
}

//______________________________________________________________________________
void TEveElement::IncImpliedSelected()
{
   // Increase element's implied-selection count. Stamp appropriately.

   if (fImpliedSelected++ == 0)
      StampColorSelection();
}

//______________________________________________________________________________
void TEveElement::DecImpliedSelected()
{
   // Decrease element's implied-selection count. Stamp appropriately.

   if (--fImpliedSelected == 0)
   {
      if (!fSelected)
         UnSelected();
      StampColorSelection();
   }
}

//______________________________________________________________________________
void TEveElement::UnSelected()
{
   // Virtual function called when both fSelected is false and
   // fImpliedSelected is 0.
   // Nothing is done in this base-class version
}

//______________________________________________________________________________
void TEveElement::HighlightElement(Bool_t state)
{
   // Set element's highlight state. Stamp appropriately.

   if (fHighlighted != state) {
      fHighlighted = state;
      if (!fHighlighted && fImpliedHighlighted == 0)
         UnHighlighted();
      fParentIgnoreCnt += (fHighlighted) ? 1 : -1;
      StampColorSelection();
   }
}

//______________________________________________________________________________
void TEveElement::IncImpliedHighlighted()
{
   // Increase element's implied-highlight count. Stamp appropriately.

   if (fImpliedHighlighted++ == 0)
      StampColorSelection();
}

//______________________________________________________________________________
void TEveElement::DecImpliedHighlighted()
{
   // Decrease element's implied-highlight count. Stamp appropriately.

   if (--fImpliedHighlighted == 0)
   {
      if (!fHighlighted)
         UnHighlighted();
      StampColorSelection();
   }
}

//______________________________________________________________________________
void TEveElement::UnHighlighted()
{
   // Virtual function called when both fHighlighted is false and
   // fImpliedHighlighted is 0.
   // Nothing is done in this base-class version
}

//______________________________________________________________________________
void TEveElement::FillImpliedSelectedSet(Set_t& impSelSet)
{
   // Populate set impSelSet with derived / dependant elements.
   //
   // If this is a TEveProjectable, the projected replicas are added
   // to the set. Thus it does not have to be reimplemented for each
   // sub-class of TEveProjected.
   //
   // Note that this also takes care of projections of TEveCompound
   // class, which is also a projectable.

   TEveProjectable* p = dynamic_cast<TEveProjectable*>(this);
   if (p)
   {
      p->AddProjectedsToSet(impSelSet);
   }
}

//______________________________________________________________________________
UChar_t TEveElement::GetSelectedLevel() const
{
   // Get selection level, needed for rendering selection and
   // highlight feedback.
   // This should go to TAtt3D.

   if (fSelected)               return 1;
   if (fImpliedSelected > 0)    return 2;
   if (fHighlighted)            return 3;
   if (fImpliedHighlighted > 0) return 4;
   return 0;
}

//______________________________________________________________________________
void TEveElement::RecheckImpliedSelections()
{
   // Call this if it is possible that implied-selection or highlight
   // has changed for this element or for implied-selection this
   // element is member of and you want to maintain consistent
   // selection state.
   // This can happen if you add elements into compounds in response
   // to user-interaction.

   if (fSelected || fImpliedSelected)
      gEve->GetSelection()->RecheckImpliedSetForElement(this);

   if (fHighlighted || fImpliedHighlighted)
      gEve->GetHighlight()->RecheckImpliedSetForElement(this);
}


/******************************************************************************/
// Stamping
/******************************************************************************/

//______________________________________________________________________________
void TEveElement::AddStamp(UChar_t bits)
{
   // Add (bitwise or) given stamps to fChangeBits.
   // Register this element to gEve as stamped.
   // This method is virtual so that sub-classes can add additional
   // actions. The base-class method should still be called (or replicated).

   fChangeBits |= bits;
   if (fDestructing == kNone) gEve->ElementStamped(this);
}

/******************************************************************************/
// List-tree icons
/******************************************************************************/

//______________________________________________________________________________
const TGPicture* TEveElement::GetListTreeIcon(Bool_t open)
{
   // Returns pointer to first listtreeicon

   // Need better solution for icon-loading/ids !!!!
   return fgListTreeIcons[open ? 7 : 0];
}

//______________________________________________________________________________
const TGPicture* TEveElement::GetListTreeCheckBoxIcon()
{
   // Returns list-tree-item check-box picture appropriate for given
   // rendering state.

   Int_t idx = 0;
   if (fRnrSelf)      idx = 2;
   if (fRnrChildren ) idx++;

   return fgRnrIcons[idx];
}

//______________________________________________________________________________
const char* TEveElement::ToString(Bool_t b)
{
   // Convert Bool_t to string - kTRUE or kFALSE.
   // Needed in WriteVizParams().

   return b ? "kTRUE" : "kFALSE";
}


/******************************************************************************/
/******************************************************************************/
// TEveElementObjectPtr
/******************************************************************************/

//______________________________________________________________________________
//
// TEveElement with external TObject as a holder of visualization data.

ClassImp(TEveElementObjectPtr);

//______________________________________________________________________________
TEveElementObjectPtr::TEveElementObjectPtr(TObject* obj, Bool_t own) :
   TEveElement (),
   TObject     (),
   fObject     (obj),
   fOwnObject  (own)
{
   // Constructor.
}

//______________________________________________________________________________
TEveElementObjectPtr::TEveElementObjectPtr(TObject* obj, Color_t& mainColor, Bool_t own) :
   TEveElement (mainColor),
   TObject     (),
   fObject     (obj),
   fOwnObject  (own)
{
   // Constructor.
}

//______________________________________________________________________________
TEveElementObjectPtr::TEveElementObjectPtr(const TEveElementObjectPtr& e) :
   TEveElement (e),
   TObject     (e),
   fObject     (0),
   fOwnObject  (e.fOwnObject)
{
   // Copy constructor.
   // If object pointed to is owned it is cloned.
   // It is assumed that the main-color has its origin in the TObject pointed to so
   // it is fixed here accordingly.

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

//______________________________________________________________________________
TEveElementObjectPtr* TEveElementObjectPtr::CloneElement() const
{
   // Clone the element via copy constructor.
   // Virtual from TEveElement.

   return new TEveElementObjectPtr(*this);
}

//______________________________________________________________________________
TObject* TEveElementObjectPtr::GetObject(const TEveException& eh) const
{
   // Return external object.
   // Virtual from TEveElement.

   if (fObject == 0)
      throw eh + "fObject not set.";
   return fObject;
}

//______________________________________________________________________________
void TEveElementObjectPtr::ExportToCINT(char* var_name)
{
   // Export external object to CINT with variable name var_name.
   // Virtual from TEveElement.

   static const TEveException eh("TEveElementObjectPtr::ExportToCINT ");

   TObject* obj = GetObject(eh);
   const char* cname = obj->IsA()->GetName();
   gROOT->ProcessLine(Form("%s* %s = (%s*)0x%lx;", cname, var_name, cname, (ULong_t)obj));
}

//______________________________________________________________________________
TEveElementObjectPtr::~TEveElementObjectPtr()
{
   // Destructor.

   if (fOwnObject)
      delete fObject;
}


/******************************************************************************/
/******************************************************************************/
// TEveElementList
/******************************************************************************/

//______________________________________________________________________________
//
// A list of TEveElements.
//
// Class of acceptable children can be limited by setting the
// fChildClass member.
//

// !!! should have two ctors (like in TEveElement), one with Color_t&
// and set fDoColor automatically, based on which ctor is called.

ClassImp(TEveElementList);

//______________________________________________________________________________
TEveElementList::TEveElementList(const char* n, const char* t, Bool_t doColor, Bool_t doTransparency) :
   TEveElement(),
   TNamed(n, t),
   TEveProjectable(),
   fColor(0),
   fChildClass(0)
{
   // Constructor.

   if (doColor) {
      fCanEditMainColor = kTRUE;
      SetMainColorPtr(&fColor);
   }
   if (doTransparency)
   {
      fCanEditMainTransparency = kTRUE;
   }
}

//______________________________________________________________________________
TEveElementList::TEveElementList(const TEveElementList& e) :
   TEveElement (e),
   TNamed      (e),
   TEveProjectable(),
   fColor      (e.fColor),
   fChildClass (e.fChildClass)
{
   // Copy constructor.
}

//______________________________________________________________________________
TEveElementList* TEveElementList::CloneElement() const
{
   // Clone the element via copy constructor.
   // Virtual from TEveElement.

   return new TEveElementList(*this);
}

//______________________________________________________________________________
Bool_t TEveElementList::AcceptElement(TEveElement* el)
{
   // Check if TEveElement el is inherited from fChildClass.
   // Virtual from TEveElement.

   if (fChildClass && ! el->IsA()->InheritsFrom(fChildClass))
      return kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
TClass* TEveElementList::ProjectedClass(const TEveProjection*) const
{
   // Virtual from TEveProjectable, returns TEveCompoundProjected class.

   return TEveElementListProjected::Class();
}


/******************************************************************************/
/******************************************************************************/
// TEveElementListProjected
/******************************************************************************/

//______________________________________________________________________________
//
// A projected element list -- required for proper propagation
// of render state to projected views.

ClassImp(TEveElementListProjected);

//______________________________________________________________________________
TEveElementListProjected::TEveElementListProjected() :
   TEveElementList("TEveElementListProjected")
{
   // Constructor.
}

//______________________________________________________________________________
void TEveElementListProjected::UpdateProjection()
{
   // This is abstract method from base-class TEveProjected.
   // No implementation.
}
