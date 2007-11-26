// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveElement
#define ROOT_TEveElement

#include "TEveUtil.h"

#include "TNamed.h"
#include "TRef.h"

class TGListTree;
class TGListTreeItem;
class TGPicture;

class TEveTrans;

/******************************************************************************/
// TEveElement
/******************************************************************************/

class TEveElement
{
   friend class TEveManager;

   TEveElement(const TEveElement&);            // Not implemented
   TEveElement& operator=(const TEveElement&); // Not implemented

public:
   class TEveListTreeInfo
   {
   public:
      TGListTree*     fTree;
      TGListTreeItem* fItem;

      TEveListTreeInfo() : fTree(0), fItem(0) {}
      TEveListTreeInfo(TGListTree* lt, TGListTreeItem* lti) : fTree(lt), fItem(lti) {}
      TEveListTreeInfo(const TEveListTreeInfo& l) : fTree(l.fTree), fItem(l.fItem) {}
      virtual ~TEveListTreeInfo() {}

      TEveListTreeInfo& operator=(const TEveListTreeInfo& l)
      { fTree = l.fTree; fItem = l.fItem; return *this; }

      bool operator==(const TEveListTreeInfo& x) const
      { return fTree == x.fTree && fItem == x.fItem; }
      bool operator<(const TEveListTreeInfo& x) const
      { return fTree == x.fTree ? fItem < x.fItem : fTree < x.fTree; }

      ClassDef(TEveListTreeInfo, 0); // Structure agregating data for a render element image in a list tree.
   };

   static const TGPicture*                      fgRnrIcons[4];
   static const TGPicture*                      fgListTreeIcons[8];

   typedef std::set<TEveListTreeInfo>           sLTI_t;
   typedef sLTI_t::iterator                     sLTI_i;
   typedef sLTI_t::reverse_iterator             sLTI_ri;

   typedef std::list<TEveElement*>              List_t;
   typedef std::list<TEveElement*>::iterator    List_i;

protected:
   // TRef     fSource;

   Bool_t   fRnrSelf;              // Render this element.
   Bool_t   fRnrChildren;          // Render children of this element.
   Color_t* fMainColorPtr;         // Pointer to main-color variable.

   sLTI_t   fItems;                // Set of list-tree-items.
   List_t   fParents;              // List of parents.

   Bool_t   fDestroyOnZeroRefCnt;  // Auto-destruct when ref-count reaches zero.
   Int_t    fDenyDestroy;          // Deny-destroy count.

   List_t   fChildren;             // List of children.

public:
   TEveElement();
   TEveElement(Color_t& main_color);
   virtual ~TEveElement();

   virtual void SetRnrElNameTitle(const Text_t* name, const Text_t* title);
   virtual const Text_t* GetRnrElName()  const;
   virtual const Text_t* GetRnrElTitle() const;

   virtual void AddParent(TEveElement* re);
   virtual void RemoveParent(TEveElement* re);
   virtual void CheckReferenceCount(const TEveException& eh="TEveElement::CheckReferenceCount ");
   virtual void CollectSceneParents(List_t& scenes);
   virtual void CollectSceneParentsFromChildren(List_t& scenes, TEveElement* parent);

   List_i BeginParents() { return fParents.begin(); }
   List_i EndParents()   { return fParents.end();   }
   Int_t  GetNParents() const { return fParents.size(); }

   List_i BeginChildren() { return fChildren.begin(); }
   List_i EndChildren()   { return fChildren.end();   }
   Int_t  GetNChildren() const { return fChildren.size(); }

   void EnableListElements (Bool_t rnr_self=kTRUE,  Bool_t rnr_children=kTRUE);  // *MENU*
   void DisableListElements(Bool_t rnr_self=kFALSE, Bool_t rnr_children=kFALSE); // *MENU*

   Bool_t GetDestroyOnZeroRefCnt() const   { return fDestroyOnZeroRefCnt; }
   void   SetDestroyOnZeroRefCnt(Bool_t d) { fDestroyOnZeroRefCnt = d; }

   Int_t  GetDenyDestroy() const { return fDenyDestroy; }
   void   IncDenyDestroy()       { ++fDenyDestroy; }
   void   DecDenyDestroy()       { if (--fDenyDestroy <= 0) CheckReferenceCount("TEveElement::DecDenyDestroy "); }

   virtual void PadPaint(Option_t* option);

   virtual TObject* GetObject(TEveException eh="TEveElement::GetObject ") const;
   virtual TObject* GetEditorObject() const { return GetObject(); }
   /*
     TRef&    GetSource() { return fSource; }
     TObject* GetSourceObject() const { return fSource.GetObject(); }
     void SetSourceObject(TObject* o) { fSource.SetObject(o); }

     void DumpSourceObject();    // *MENU*
     void InspectSourceObject(); // *MENU*
   */

   // --------------------------------

   virtual Int_t ExpandIntoListTree(TGListTree* ltree, TGListTreeItem* parent);
   virtual Int_t DestroyListSubTree(TGListTree* ltree, TGListTreeItem* parent);

   virtual TGListTreeItem* AddIntoListTree(TGListTree* ltree,
                                           TGListTreeItem* parent_lti);
   virtual TGListTreeItem* AddIntoListTree(TGListTree* ltree,
                                           TEveElement* parent);
   virtual TGListTreeItem* AddIntoListTrees(TEveElement* parent);

   virtual Bool_t          RemoveFromListTree(TGListTree* ltree,
                                              TGListTreeItem* parent_lti);
   virtual Int_t           RemoveFromListTrees(TEveElement* parent);

   virtual sLTI_i          FindItem(TGListTree* ltree);
   virtual sLTI_i          FindItem(TGListTree* ltree,
                                    TGListTreeItem* parent_lti);
   virtual TGListTreeItem* FindListTreeItem(TGListTree* ltree);
   virtual TGListTreeItem* FindListTreeItem(TGListTree* ltree,
                                            TGListTreeItem* parent_lti);

   virtual Int_t GetNItems() const { return fItems.size(); }
   virtual void  UpdateItems();

   void SpawnEditor();                          // *MENU*
   virtual void ExportToCINT(Text_t* var_name); // *MENU*

   virtual Bool_t AcceptElement(TEveElement* /*el*/) { return kTRUE; }

   virtual TGListTreeItem* AddElement(TEveElement* el);
   virtual void RemoveElement(TEveElement* el);
   virtual void RemoveElementLocal(TEveElement* el);
   virtual void RemoveElements();
   virtual void RemoveElementsLocal();

   virtual void Destroy();                      // *MENU*
   virtual void DestroyElements();              // *MENU*

   virtual Bool_t HandleElementPaste(TEveElement* el);
   virtual void   ElementChanged(Bool_t update_scenes=kTRUE, Bool_t redraw=kFALSE);

   virtual Bool_t CanEditRnrElement()    { return kTRUE; }
   virtual Bool_t GetRnrSelf()     const { return fRnrSelf; }
   virtual Bool_t GetRnrChildren() const { return fRnrChildren; }
   virtual void   SetRnrSelf(Bool_t rnr);
   virtual void   SetRnrChildren(Bool_t rnr);
   virtual void   SetRnrState(Bool_t rnr);

   virtual Bool_t CanEditMainColor()        { return kFALSE; }
   Color_t* GetMainColorPtr()               { return fMainColorPtr; }
   void     SetMainColorPtr(Color_t* color) { fMainColorPtr = color; }

   virtual Color_t GetMainColor() const { return fMainColorPtr ? *fMainColorPtr : 0; }
   virtual void    SetMainColor(Color_t color);
   void    SetMainColor(Pixel_t pixel);

   virtual Bool_t  CanEditMainTransparency()    { return kFALSE; }
   virtual UChar_t GetMainTransparency() const  { return 0; }
   virtual void    SetMainTransparency(UChar_t) {}

   virtual Bool_t  CanEditMainHMTrans() { return kFALSE; }
   virtual TEveTrans* PtrMainHMTrans()     { return 0; }

   static  const TGPicture* GetCheckBoxPicture(Bool_t rnrElement, Bool_t rnrDaughter);
   virtual const TGPicture* GetListTreeIcon() { return fgListTreeIcons[0]; }

   ClassDef(TEveElement, 1); // Base class for TEveUtil visualization elements, providing hierarchy management, rendering control and list-tree item management.
};


/******************************************************************************/
// TEveElementObjectPtr
/******************************************************************************/

class TEveElementObjectPtr : public TEveElement,
                             public TObject
{
   TEveElementObjectPtr(const TEveElementObjectPtr&);            // Not implemented
   TEveElementObjectPtr& operator=(const TEveElementObjectPtr&); // Not implemented

protected:
   TObject* fObject;     // External object holding the visual data.
   Bool_t   fOwnObject;  // Is object owned / should be deleted on destruction.

public:
   TEveElementObjectPtr(TObject* obj, Bool_t own=kTRUE);
   TEveElementObjectPtr(TObject* obj, Color_t& mainColor, Bool_t own=kTRUE);
   virtual ~TEveElementObjectPtr();

   virtual TObject* GetObject(TEveException eh="TEveElementObjectPtr::GetObject ") const;
   virtual void     ExportToCINT(Text_t* var_name);

   Bool_t GetOwnObject() const   { return fOwnObject; }
   void   SetOwnObject(Bool_t o) { fOwnObject = o; }

   ClassDef(TEveElementObjectPtr, 1); // TEveElement with external TObject as a holder of visualization data.
};


/******************************************************************************/
// TEveElementList
/******************************************************************************/

class TEveElementList : public TEveElement,
                        public TNamed
{
protected:
   Color_t   fColor;       // Color of the object.
   Bool_t    fDoColor;     // Should serve fColor as the main color of the object.
   TClass   *fChildClass;  // Class of acceptable children, others are rejected.

public:
   TEveElementList(const Text_t* n="TEveElementList", const Text_t* t="",
                   Bool_t doColor=kFALSE);
   virtual ~TEveElementList() {}

   virtual Bool_t CanEditMainColor()  { return fDoColor; }

   TClass* GetChildClass() const { return fChildClass; }
   void SetChildClass(TClass* c) { fChildClass = c; }

   virtual Bool_t AcceptElement(TEveElement* el);

   ClassDef(TEveElementList, 1); // List of TEveElement objects with a possibility to limit the class of accepted elements.
};

#endif
