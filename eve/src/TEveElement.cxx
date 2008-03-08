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
#include "TEveTrans.h"
#include "TEveManager.h"
#include "TEveSelection.h"
#include "TEveProjectionBases.h"

#include "TGeoMatrix.h"

#include "TClass.h"
#include "TPRegexp.h"
#include "TROOT.h"
#include "TColor.h"
#include "TCanvas.h"
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
const TGPicture* TEveElement::fgListTreeIcons[8] = { 0 };

//______________________________________________________________________________
TEveElement::TEveElement() :
   fParents             (),
   fChildren            (),
   fDestroyOnZeroRefCnt (kTRUE),
   fDenyDestroy         (0),
   fRnrSelf             (kTRUE),
   fRnrChildren         (kTRUE),
   fCanEditMainTrans    (kFALSE),
   fMainColorPtr        (0),
   fMainTrans           (0),
   fItems               (),
   fUserData            (0),
   fPickable            (kFALSE),
   fSelected            (kFALSE),
   fHighlighted         (kFALSE),
   fImpliedSelected     (0),
   fImpliedHighlighted  (0),
   fChangeBits          (0),
   fDestructing         (kFALSE)
{
   // Default contructor.
}

//______________________________________________________________________________
TEveElement::TEveElement(Color_t& main_color) :
   fParents             (),
   fChildren            (),
   fDestroyOnZeroRefCnt (kTRUE),
   fDenyDestroy         (0),
   fRnrSelf             (kTRUE),
   fRnrChildren         (kTRUE),
   fCanEditMainTrans    (kFALSE),
   fMainColorPtr        (&main_color),
   fMainTrans           (0),
   fItems               (),
   fUserData            (0),
   fPickable            (kFALSE),
   fSelected            (kFALSE),
   fHighlighted         (kFALSE),
   fImpliedSelected     (0),
   fImpliedHighlighted  (0),
   fChangeBits          (0),
   fDestructing         (kFALSE)
{
   // Constructor.
}

//______________________________________________________________________________
TEveElement::~TEveElement()
{
   // Destructor.

   fDestructing = kTRUE;

   RemoveElementsInternal();

   for (List_i p=fParents.begin(); p!=fParents.end(); ++p)
   {
      (*p)->RemoveElementLocal(this);
      (*p)->fChildren.remove(this);
   }
   fParents.clear();

   for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
      i->fTree->DeleteItem(i->fItem);

   delete fMainTrans;
}

/******************************************************************************/

//______________________________________________________________________________
const Text_t* TEveElement::GetElementName() const
{
   // Virtual function for retrieveing name of the element.
   // Here we attempt to cast the assigned object into TNamed and call
   // GetName() there.

   static const TEveException eh("TEveElement::GetElementName ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   return named ? named->GetName() : "<no-name>";
}

//______________________________________________________________________________
const Text_t*  TEveElement::GetElementTitle() const
{
   // Virtual function for retrieveing title of the render-element.
   // Here we attempt to cast the assigned object into TNamed and call
   // GetTitle() there.

   static const TEveException eh("TEveElement::GetElementTitle ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   return named ? named->GetTitle() : "<no-title>";
}

//______________________________________________________________________________
void TEveElement::SetElementName(const Text_t* name)
{
   // Virtual function for setting of name of an element.
   // Here we attempt to cast the assigned object into TNamed and call
   // SetName() there.

   static const TEveException eh("TEveElement::SetElementName ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   if (named)
      named->SetName(name);
}

//______________________________________________________________________________
void TEveElement::SetElementTitle(const Text_t* title)
{
   // Virtual function for setting of title of an element.
   // Here we attempt to cast the assigned object into TNamed and call
   // SetTitle() there.

   static const TEveException eh("TEveElement::SetElementTitle ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   if (named)
      named->SetTitle(title);
}

//______________________________________________________________________________
void TEveElement::SetElementNameTitle(const Text_t* name, const Text_t* title)
{
   // Virtual function for setting of name and title of render element.
   // Here we attempt to cast the assigned object into TNamed and call
   // SetNameTitle() there.

   static const TEveException eh("TEveElement::SetElementNameTitle ");

   TNamed* named = dynamic_cast<TNamed*>(GetObject(eh));
   if (named)
      named->SetNameTitle(name, title);
}

/******************************************************************************/

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

   UInt_t parent_cnt = 0, item_cnt = 0;
   if (fSelected)
   {
      ++parent_cnt;
      item_cnt += gEve->GetSelection()->GetNItems();
   }
   if (fHighlighted)
   {
      ++parent_cnt;
      item_cnt += gEve->GetHighlight()->GetNItems();
   }

   if(fParents.size() <= parent_cnt && fItems.size() <= item_cnt &&
      fDenyDestroy    <= 0          && fDestroyOnZeroRefCnt)
   {
      if (gDebug > 0)
         Info(eh, Form("auto-destructing '%s' on zero reference count.", GetElementName()));

      gEve->PreDeleteElement(this);
      delete this;
   }
}

//______________________________________________________________________________
void TEveElement::CollectSceneParents(List_t& scenes)
{
   // Collect all parents of class TEveScene. This is needed to
   // automatically detect which scenes need to be updated.
   //
   // Overriden in TEveScene to include itself and return.

   for(List_i p=fParents.begin(); p!=fParents.end(); ++p)
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
   // parent_lti.

   static const TEveException eh("TEveElement::AddIntoListTree ");

   TGListTreeItem* item = new TEveListTreeItem(this);
   ltree->AddItem(parent_lti, item);

   fItems.insert(TEveListTreeInfo(ltree, item));
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
      if (parent_lti == 0) CheckReferenceCount(eh);
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
Int_t TEveElement::RemoveFromListTrees(TEveElement* parent)
{
   // Remove element from all list-trees where 'parent' is the
   // user-data of the list-tree-item.

   Int_t count = 0;

   sLTI_i i  = fItems.begin();
   while (i != fItems.end())
   {
      sLTI_i j = i++;
      TGListTreeItem *plti = j->fItem->GetParent();
      if (plti != 0 && (TEveElement*) plti->GetUserData() == parent)
      {
         DestroyListSubTree(j->fTree, j->fItem);
         j->fTree->DeleteItem(j->fItem);
         j->fTree->ClearViewPort();
         fItems.erase(j);
         ++count;
      }
   }

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

//______________________________________________________________________________
void TEveElement::UpdateItems()
{
   // Update list-tree-items representing this element.

   static const TEveException eh("TEveElement::UpdateItems ");

   for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
      i->fTree->ClearViewPort();
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
void TEveElement::ExportToCINT(Text_t* var_name)
{
   // Export render-element to CINT with variable name var_name.

   const char* cname = IsA()->GetName();
   gROOT->ProcessLine(Form("%s* %s = (%s*)0x%lx;", cname, var_name, cname, this));
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

/******************************************************************************/

//______________________________________________________________________________
void TEveElement::SetRnrSelf(Bool_t rnr)
{
   // Set render state of this element, i.e. if it will be published
   // on next scene update pass.

   if (SingleRnrState())
   {
      SetRnrState(rnr);
      return;
   }

   if (rnr != fRnrSelf)
   {
      fRnrSelf = rnr;
      UpdateItems();
   }
}

//______________________________________________________________________________
void TEveElement::SetRnrChildren(Bool_t rnr)
{
   // Set render state of this element's children, i.e. if they will
   // be published on next scene update pass.

   if (SingleRnrState())
   {
      SetRnrState(rnr);
      return;
   }

   if (rnr != fRnrChildren)
   {
      fRnrChildren = rnr;
      UpdateItems();
   }
}

//______________________________________________________________________________
void TEveElement::SetRnrState(Bool_t rnr)
{
   // Set render state of this element and of its children to the same
   // value.

   if (fRnrSelf != rnr || fRnrChildren != rnr)
   {
      fRnrSelf = fRnrChildren = rnr;
      UpdateItems();
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElement::SetMainColor(Color_t color)
{
   // Set main color of the element.
   // List-tree-items are updated.

   Color_t oldcol = GetMainColor();

   // !!!!! WTF is this? At least should be moved somewhere else ...
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i) {
      if ((*i)->GetMainColor() == oldcol) (*i)->SetMainColor(color);
   }

   if (fMainColorPtr) {
      *fMainColorPtr = color;
      StampColorSelection();
   }
}

//______________________________________________________________________________
void TEveElement::SetMainColor(Pixel_t pixel)
{
   // Convert pixel to Color_t and call the above function.

   SetMainColor(Color_t(TColor::GetColor(pixel)));
}

/******************************************************************************/

//______________________________________________________________________________
TEveTrans* TEveElement::PtrMainTrans()
{
   // Return pointer to main transformation. It is created if not yet
   // existing.

   if (!fMainTrans)
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
   fChildren.push_back(el);
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
   fChildren.remove(el);
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
   fChildren.clear();
}

//______________________________________________________________________________
void TEveElement::RemoveElements()
{
   // Remove all elements. This assumes removing of all elements can
   // be done more efficiently then looping over them and removing
   // them one by one.

   if ( ! fChildren.empty())
   {
      RemoveElementsInternal();
      ElementChanged();
   }
}

//______________________________________________________________________________
void TEveElement::RemoveElementsLocal()
{
   // Perform additional local removal of all elements.
   // See comment to RemoveelementLocal(TEveElement*).
}

/******************************************************************************/

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

/******************************************************************************/

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
void TEveElement::Destroy()
{
   // Destroy this element.

   static const TEveException eh("TEveElement::Destroy ");

   if (fDenyDestroy > 0)
      throw(eh + "this element '%s' is protected against destruction.", GetElementName());

   gEve->PreDeleteElement(this);
   delete this;
   gEve->Redraw3D();
}

//______________________________________________________________________________
void TEveElement::DestroyElements()
{
   // Destroy all children of this element.

   static const TEveException eh("TEveElement::DestroyElements ");

   while ( ! fChildren.empty()) {
      TEveElement* c = fChildren.front();
      if (c->fDenyDestroy <= 0)
      {
         try {
            c->Destroy();
         }
         catch (TEveException exc) {
            Warning(eh, Form("element destruction failed: '%s'.", exc.Data()));
            RemoveElement(c);
         }
      }
      else
      {
         if (gDebug > 0)
            Info(eh, Form("element '%s' is protected agains destruction, removing locally.",
			  c->GetElementName()));
         RemoveElement(c);
      }
   }

   gEve->Redraw3D();
}

/******************************************************************************/

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
void TEveElement::SelectElement(Bool_t state)
{
   // Set element's selection state. Stamp appropriately.

   if (fSelected != state) {
      fSelected = state;
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
      StampColorSelection();
}

//______________________________________________________________________________
void TEveElement::HighlightElement(Bool_t state)
{
   // Set element's highlight state. Stamp appropriately.

   if (fHighlighted != state) {
      fHighlighted = state;
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
      StampColorSelection();
}

//______________________________________________________________________________
void TEveElement::FillImpliedSelectedSet(Set_t& impSelSet)
{
   // Populate set impSelSet with derived / dependant elements.

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

/******************************************************************************/
// Stamping
/******************************************************************************/

//______________________________________________________________________________
void TEveElement::SetStamp(UChar_t bits)
{
   // Set fChangeBits to bits.
   // Register this element to gEve as stamped.

   fChangeBits = bits;
   if (!fDestructing) gEve->ElementStamped(this);
}

//______________________________________________________________________________
void TEveElement::AddStamp(UChar_t bits)
{
   // Add (bitwise or) given stamps to fChangeBits.
   // Register this element to gEve as stamped.

   fChangeBits |= bits;
   if (!fDestructing) gEve->ElementStamped(this);
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
   TEveElement(),
   fObject(obj),
   fOwnObject(own)
{
   // Constructor.
}

//______________________________________________________________________________
TEveElementObjectPtr::TEveElementObjectPtr(TObject* obj, Color_t& mainColor, Bool_t own) :
   TEveElement(mainColor),
   fObject(obj),
   fOwnObject(own)
{
   // Constructor.
}

//______________________________________________________________________________
TObject* TEveElementObjectPtr::GetObject(const TEveException& eh) const
{
   // Return external object.
   // Virtual from TEveElement.

   if(fObject == 0)
      throw(eh + "fObject not set.");
   return fObject;
}

//______________________________________________________________________________
void TEveElementObjectPtr::ExportToCINT(Text_t* var_name)
{
   // Export external object to CINT with variable name var_name.
   // Virtual from TEveElement.

   static const TEveException eh("TEveElementObjectPtr::ExportToCINT ");

   TObject* obj = GetObject(eh);
   const char* cname = obj->IsA()->GetName();
   gROOT->ProcessLine(Form("%s* %s = (%s*)0x%lx;", cname, var_name, cname, obj));
}

//______________________________________________________________________________
TEveElementObjectPtr::~TEveElementObjectPtr()
{
   // Destructor.

   if(fOwnObject)
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
TEveElementList::TEveElementList(const Text_t* n, const Text_t* t, Bool_t doColor) :
   TEveElement(),
   TNamed(n, t),
   fColor(0),
   fDoColor(doColor),
   fChildClass(0)
{
   // Constructor.

   if(fDoColor) {
      SetMainColorPtr(&fColor);
   }
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
