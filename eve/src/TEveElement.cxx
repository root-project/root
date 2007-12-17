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
#include "TEveManager.h"
#include "TEveGedEditor.h"

#include "TColor.h"
#include "TCanvas.h"
#include "TGListTree.h"
#include "TGPicture.h"

#include <algorithm>

//______________________________________________________________________________
//
// Structure holding information about TGListTree and TGListTreeItem
// that represents given TEveElement. This needed because each element
// can appear in several list-trees as well as several times in the
// same list-tree.

ClassImp(TEveElement::TEveListTreeInfo)

//______________________________________________________________________________
// TEveElement
//
// Base class for TEveUtil visualization elements, providing hierarchy
// management, rendering control and list-tree item management.

ClassImp(TEveElement)

//______________________________________________________________________________
const TGPicture* TEveElement::fgRnrIcons[4] = { 0 };
const TGPicture* TEveElement::fgListTreeIcons[8] = { 0 };

//______________________________________________________________________________
TEveElement::TEveElement() :
   fRnrSelf             (kTRUE),
   fRnrChildren         (kTRUE),
   fMainColorPtr        (0),
   fItems               (),
   fParents             (),
   fDestroyOnZeroRefCnt (kTRUE),
   fDenyDestroy         (0),
   fChildren            ()
{
   // Default contructor.
}

//______________________________________________________________________________
TEveElement::TEveElement(Color_t& main_color) :
   fRnrSelf             (kTRUE),
   fRnrChildren         (kTRUE),
   fMainColorPtr        (&main_color),
   fItems               (),
   fParents             (),
   fDestroyOnZeroRefCnt (kTRUE),
   fDenyDestroy         (0),
   fChildren            ()
{
   // Constructor.
}

//______________________________________________________________________________
TEveElement::~TEveElement()
{
   // Destructor.

   RemoveElements();

   for (List_i p=fParents.begin(); p!=fParents.end(); ++p)
   {
      (*p)->RemoveElementLocal(this);
      (*p)->fChildren.remove(this);
   }
   fParents.clear();

   for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
      i->fTree->DeleteItem(i->fItem);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElement::SetRnrElNameTitle(const Text_t* name, const Text_t* title)
{
   // Virtual function for setting of name and title of render element.
   // Here we attempt to cast the assigned object into TNamed and call
   // SetNameTitle() there.

   TNamed* named = dynamic_cast<TNamed*>(GetObject());
   if (named)
      named->SetNameTitle(name, title);
}

//______________________________________________________________________________
const Text_t* TEveElement::GetRnrElName() const
{
   // Virtual function for retrieveing name of the render-element.
   // Here we attempt to cast the assigned object into TNamed and call
   // GetName() there.

   TObject* named = dynamic_cast<TObject*>(GetObject());
   return named ? named->GetName() : "<no-name>";
}

//______________________________________________________________________________
const Text_t*  TEveElement::GetRnrElTitle() const
{
   // Virtual function for retrieveing title of the render-element.
   // Here we attempt to cast the assigned object into TNamed and call
   // GetTitle() there.

   TObject* named = dynamic_cast<TObject*>(GetObject());
   return named ? named->GetTitle() : "<no-title>";
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

   static const TEveException eH("TEveElement::RemoveParent ");

   fParents.remove(re);
   CheckReferenceCount(eH);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElement::CheckReferenceCount(const TEveException& eh)
{
   // Check external references to this and eventually auto-destruct
   // the render-element.

   if(fParents.empty()   &&  fItems.empty()         &&
      fDenyDestroy <= 0  &&  fDestroyOnZeroRefCnt)
   {
      if (gDebug > 0)
         Info(eh, Form("auto-destructing '%s' on zero reference count.", GetRnrElName()));

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
void TEveElement::CollectSceneParentsFromChildren(List_t& scenes, TEveElement* parent)
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
   // RnrEl can be inserted in a list-tree several times, thus we can not
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

   static const TEveException eH("TEveElement::AddIntoListTree ");

   TObject* tobj = GetObject(eH);
   TGListTreeItem* item = ltree->AddItem(parent_lti, tobj->GetName(), this,
                                         0, 0, kTRUE);
   item->SetCheckBoxPictures(GetCheckBoxPicture(1, fRnrChildren),
                             GetCheckBoxPicture(0, fRnrChildren));

   item->SetPictures(GetListTreeIcon(),GetListTreeIcon());
   item->CheckItem(fRnrSelf);

   if (fMainColorPtr != 0) item->SetColor(GetMainColor());
   item->SetTipText(tobj->GetTitle());

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

   static const TEveException eH("TEveElement::RemoveFromListTree ");

   sLTI_i i = FindItem(ltree, parent_lti);
   if (i != fItems.end()) {
      DestroyListSubTree(ltree, i->fItem);
      ltree->DeleteItem(i->fItem);
      ltree->ClearViewPort();
      fItems.erase(i);
      if (parent_lti == 0) CheckReferenceCount(eH);
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

   static const TEveException eH("TEveElement::UpdateItems ");

   TObject* tobj = GetObject(eH);

   for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i) {
      i->fItem->Rename(tobj->GetName());
      i->fItem->SetTipText(tobj->GetTitle());
      i->fItem->CheckItem(fRnrSelf);
      if (fMainColorPtr != 0) i->fItem->SetColor(GetMainColor());
      i->fTree->ClearViewPort();
   }
}

/******************************************************************************/

//______________________________________________________________________________
TObject* TEveElement::GetObject(TEveException eh) const
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

   if (GetRnrSelf() && GetObject())
      GetObject()->Paint(option);


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

   if (rnr != fRnrSelf)
   {
      fRnrSelf = rnr;

      for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
      {
         if (i->fItem->IsChecked() != rnr) {
            i->fItem->SetCheckBoxPictures(GetCheckBoxPicture(1, fRnrChildren),
                                          GetCheckBoxPicture(0, fRnrChildren));
            i->fItem->CheckItem(fRnrSelf);
            i->fTree->ClearViewPort();
         }
      }
   }
}

//______________________________________________________________________________
void TEveElement::SetRnrChildren(Bool_t rnr)
{
   // Set render state of this element's children, i.e. if they will
   // be published on next scene update pass.

   if (rnr != fRnrChildren)
   {
      fRnrChildren = rnr;

      for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
      {
         i->fItem->SetCheckBoxPictures(GetCheckBoxPicture(fRnrSelf, fRnrChildren),
                                       GetCheckBoxPicture(fRnrSelf, fRnrChildren));
         i->fTree->ClearViewPort();
      }
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

      for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i)
      {
         i->fItem->SetCheckBoxPictures(GetCheckBoxPicture(1,1), GetCheckBoxPicture(0,0));
         i->fItem->CheckItem(fRnrSelf);
         i->fTree->ClearViewPort();
      }
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveElement::SetMainColor(Color_t color)
{
   // Set main color of the render-element.
   // List-tree-items are updated.

   Color_t oldcol = GetMainColor();
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i) {
      if ((*i)->GetMainColor() == oldcol) (*i)->SetMainColor(color);
   }

   if (fMainColorPtr) {
      *fMainColorPtr = color;
      for (sLTI_i i=fItems.begin(); i!=fItems.end(); ++i) {
         if (i->fItem->GetColor() != color) {
            i->fItem->SetColor(GetMainColor());
            i->fTree->ClearViewPort();
         }
      }
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
TGListTreeItem* TEveElement::AddElement(TEveElement* el)
{
   // Add el to the list of children.

   static const TEveException eH("TEveElement::AddElement ");

   if ( ! AcceptElement(el))
      throw(eH + Form("parent '%s' rejects '%s'.",
                      GetRnrElName(), el->GetRnrElName()));

   el->AddParent(this);
   fChildren.push_back(el);
   TGListTreeItem* ret = el->AddIntoListTrees(this);
   ElementChanged();
   return ret;
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
void TEveElement::RemoveElements()
{
   // Remove all elements. This assumes removing of all elements can be
   // done more efficiently then looping over them and removing one by
   // one.

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
   ElementChanged();
}

//______________________________________________________________________________
void TEveElement::RemoveElementsLocal()
{
   // Perform additional local removal of all elements.
   // See comment to RemoveelementLocal(TEveElement*).
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

   static const TEveException eH("TEveElement::Destroy ");

   if (fDenyDestroy > 0)
      throw(eH + "this element '%s' is protected against destruction.", GetRnrElName());

   gEve->PreDeleteElement(this);
   delete this;
   gEve->Redraw3D();
}

//______________________________________________________________________________
void TEveElement::DestroyElements()
{
   // Destroy all children of this element.

   static const TEveException eH("TEveElement::DestroyElements ");

   while ( ! fChildren.empty()) {
      TEveElement* c = fChildren.front();
      if (c->fDenyDestroy <= 0)
      {
         try {
            c->Destroy();
         }
         catch (TEveException exc) {
            Warning(eH, Form("element destruction failed: '%s'.", exc.Data()));
            RemoveElement(c);
         }
      }
      else
      {
         if (gDebug > 0)
            Info(eH, Form("element '%s' is protected agains destruction, removin locally.", c->GetRnrElName()));

         RemoveElement(c);
      }
   }
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

   if (update_scenes)
      gEve->ElementChanged(this);
   if (redraw)
      gEve->Redraw3D();
}

/******************************************************************************/
// Statics
/******************************************************************************/

//______________________________________________________________________________
const TGPicture*
TEveElement::GetCheckBoxPicture(Bool_t rnrSelf, Bool_t rnrDaughters)
{
   // Returns list-tree-item check-box picture appropriate for given
   // rendering state.

   Int_t idx = 0;
   if (rnrSelf)       idx = 2;
   if (rnrDaughters ) idx++;

   return fgRnrIcons[idx];
}


//______________________________________________________________________________
// TEveElementObjectPtr
//
// TEveElement with external TObject as a holder of visualization data.

ClassImp(TEveElementObjectPtr)

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
TObject* TEveElementObjectPtr::GetObject(TEveException eh) const
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

   static const TEveException eH("TEveElementObjectPtr::ExportToCINT ");

   TObject* obj = GetObject(eH);
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

//______________________________________________________________________________
// TEveElementList
//
// A list of TEveElements.
//
// Class of acceptable children can be limited by setting the
// fChildClass member.
//

// !!! should have two ctors (like in TEveElement), one with Color_t&
// and set fDoColor automatically, based on which ctor is called.

ClassImp(TEveElementList)

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
