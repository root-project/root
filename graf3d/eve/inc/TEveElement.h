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

class TEveCompound;
class TEveTrans;
class TGeoMatrix;

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

   typedef std::set<TEveElement*>               Set_t;
   typedef std::set<TEveElement*>::iterator     Set_i;

protected:
   TEveCompound    *fCompound;             //  Compound this object belongs to.

   List_t           fParents;              //  List of parents.
   List_t           fChildren;             //  List of children.

   Bool_t           fDestroyOnZeroRefCnt;  //  Auto-destruct when ref-count reaches zero.
   Int_t            fDenyDestroy;          //  Deny-destroy count.

   Bool_t           fRnrSelf;              //  Render this element.
   Bool_t           fRnrChildren;          //  Render children of this element.
   Bool_t           fCanEditMainTrans;     //  Allow editing of main transformation.

   Color_t         *fMainColorPtr;         //  Pointer to main-color variable.
   TEveTrans       *fMainTrans;            //  Pointer to main transformation matrix.

   sLTI_t           fItems;                //! Set of list-tree-items.

   TRef             fSource;               //  External object that is represented by this element.
   void            *fUserData;             //! Externally assigned and controlled user data.

   virtual void RemoveElementsInternal();

public:
   TEveElement();
   TEveElement(Color_t& main_color);
   virtual ~TEveElement();

   virtual const Text_t* GetElementName()  const;
   virtual const Text_t* GetElementTitle() const;
   virtual void SetElementName (const Text_t* name);
   virtual void SetElementTitle(const Text_t* title);
   virtual void SetElementNameTitle(const Text_t* name, const Text_t* title);

   TEveElement*  GetMaster();
   TEveCompound* GetCompound()                { return fCompound; }
   void          SetCompound(TEveCompound* c) { fCompound = c;    }

   virtual void AddParent(TEveElement* re);
   virtual void RemoveParent(TEveElement* re);
   virtual void CheckReferenceCount(const TEveException& eh="TEveElement::CheckReferenceCount ");
   virtual void CollectSceneParents(List_t& scenes);
   virtual void CollectSceneParentsFromChildren(List_t& scenes,
                                                TEveElement* parent);

   List_i BeginParents() { return fParents.begin(); }
   List_i EndParents()   { return fParents.end();   }
   Int_t  GetNParents() const { return fParents.size(); }

   List_i BeginChildren() { return fChildren.begin(); }
   List_i EndChildren()   { return fChildren.end();   }
   Int_t  GetNChildren() const { return fChildren.size(); }

   Bool_t       HasChild(TEveElement* el);
   TEveElement* FindChild(const TString& name, const TClass* cls=0);
   TEveElement* FindChild(TPRegexp& regexp, const TClass* cls=0);
   Int_t        FindChildren(List_t& matches, const TString&  name, const TClass* cls=0);
   Int_t        FindChildren(List_t& matches, TPRegexp& regexp, const TClass* cls=0);

   void EnableListElements (Bool_t rnr_self=kTRUE,  Bool_t rnr_children=kTRUE);  // *MENU*
   void DisableListElements(Bool_t rnr_self=kFALSE, Bool_t rnr_children=kFALSE); // *MENU*

   Bool_t GetDestroyOnZeroRefCnt() const   { return fDestroyOnZeroRefCnt; }
   void   SetDestroyOnZeroRefCnt(Bool_t d) { fDestroyOnZeroRefCnt = d; }

   Int_t  GetDenyDestroy() const { return fDenyDestroy; }
   void   IncDenyDestroy()       { ++fDenyDestroy; }
   void   DecDenyDestroy()       { if (--fDenyDestroy <= 0) CheckReferenceCount("TEveElement::DecDenyDestroy "); }

   virtual void PadPaint(Option_t* option);

   virtual TObject* GetObject      (const TEveException& eh="TEveElement::GetObject ") const;
   virtual TObject* GetEditorObject(const TEveException& eh="TEveElement::GetEditorObject ") const { return GetObject(eh); }
   virtual TObject* GetRenderObject(const TEveException& eh="TEveElement::GetRenderObject ") const { return GetObject(eh); }

   // --------------------------------

   virtual void ExpandIntoListTree(TGListTree* ltree, TGListTreeItem* parent);
   virtual void DestroyListSubTree(TGListTree* ltree, TGListTreeItem* parent);

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

   virtual Bool_t AcceptElement(TEveElement* el);

   virtual void AddElement(TEveElement* el);
   virtual void RemoveElement(TEveElement* el);
   virtual void RemoveElementLocal(TEveElement* el);
   virtual void RemoveElements();
   virtual void RemoveElementsLocal();

   virtual void Destroy();                      // *MENU*
   virtual void DestroyElements();              // *MENU*

   virtual Bool_t HandleElementPaste(TEveElement* el);
   virtual void   ElementChanged(Bool_t update_scenes=kTRUE, Bool_t redraw=kFALSE);

   virtual Bool_t CanEditElement() const { return kTRUE; }
   virtual Bool_t SingleRnrState() const { return kFALSE; }
   virtual Bool_t GetRnrSelf()     const { return fRnrSelf; }
   virtual Bool_t GetRnrChildren() const { return fRnrChildren; }
   virtual Bool_t GetRnrState()    const { return fRnrSelf && fRnrChildren; }
   virtual void   SetRnrSelf(Bool_t rnr);
   virtual void   SetRnrChildren(Bool_t rnr);
   virtual void   SetRnrState(Bool_t rnr);

   virtual Bool_t CanEditMainColor() const  { return kFALSE; }
   Color_t* GetMainColorPtr()               { return fMainColorPtr; }
   void     SetMainColorPtr(Color_t* color) { fMainColorPtr = color; }

   virtual Bool_t  HasMainColor() const { return fMainColorPtr != 0; }
   virtual Color_t GetMainColor() const { return fMainColorPtr ? *fMainColorPtr : 0; }
   virtual void    SetMainColor(Color_t color);
   void            SetMainColorPixel(Pixel_t pixel);
   void            SetMainColorRGB(UChar_t r, UChar_t g, UChar_t b);
   void            SetMainColorRGB(Float_t r, Float_t g, Float_t b);

   virtual Bool_t  CanEditMainTransparency() const { return kFALSE; }
   virtual UChar_t GetMainTransparency()     const { return 0; }
   virtual void    SetMainTransparency(UChar_t) {}

   virtual Bool_t     CanEditMainTrans() const { return fCanEditMainTrans; }
   virtual Bool_t     HasMainTrans()     const { return fMainTrans != 0;   }
   virtual TEveTrans* PtrMainTrans();
   virtual TEveTrans& RefMainTrans();
   virtual void       InitMainTrans(Bool_t can_edit=kTRUE);
   virtual void       DestroyMainTrans();

   virtual void SetTransMatrix(Double_t* carr);
   virtual void SetTransMatrix(const TGeoMatrix& mat);

   TRef&    GetSource()                 { return fSource; }
   TObject* GetSourceObject()     const { return fSource.GetObject(); }
   void     SetSourceObject(TObject* o) { fSource.SetObject(o); }
   /*
     void DumpSourceObject();    // *MENU*
     void InspectSourceObject(); // *MENU*
   */

   void* GetUserData() const { return fUserData; }
   void  SetUserData(void* ud) { fUserData = ud; }

   // Selection state and management
   //--------------------------------
protected:
   Bool_t  fPickable;
   Bool_t  fSelected;             //!
   Bool_t  fHighlighted;          //!
   Short_t fImpliedSelected;      //!
   Short_t fImpliedHighlighted;   //!

public:
   typedef void (TEveElement::* Select_foo)      (Bool_t);
   typedef void (TEveElement::* ImplySelect_foo) ();

   Bool_t IsPickable()    const { return fPickable; }
   void   SetPickable(Bool_t p) { fPickable = p; }

   void SelectElement(Bool_t state);
   void IncImpliedSelected();
   void DecImpliedSelected();

   void HighlightElement(Bool_t state);
   void IncImpliedHighlighted();
   void DecImpliedHighlighted();

   virtual void FillImpliedSelectedSet(Set_t& impSelSet);

   virtual UChar_t GetSelectedLevel() const;

   // Change-stamping and change bits
   //---------------------------------

   enum EChangeBits
   {
      kCBColorSelection =  1, // Main color or select/hilite state changed.
      kCBTransBBox      =  2, // Transformation matrix or bounding-box changed.
      kCBObjProps       =  4  // Object changed, requires dropping its display-lists.
      // kCBElementAdded   =  8, // Element was added to a new parent.
      // kCBElementRemoved = 16  // Element was removed from a parent.

      // Deletions are handled in a special way in TEveManager::PreDeleteElement().
   };

protected:
   UChar_t      fChangeBits;
   Bool_t       fDestructing;

public:
   void StampColorSelection() { AddStamp(kCBColorSelection); }
   void StampTransBBox()      { AddStamp(kCBTransBBox); }
   void StampObjProps()       { AddStamp(kCBObjProps); }
   // void StampElementAdded()   { AddStamp(kCBElementAdded); }
   // void StampElementRemoved() { AddStamp(kCBElementRemoved); }
   void ClearStamps()         { fChangeBits = 0; }
   void SetStamp(UChar_t bits);
   void AddStamp(UChar_t bits);

   UChar_t GetChangeBits() const { return fChangeBits; }


   // List-tree icons
   //-----------------

   virtual const TGPicture* GetListTreeIcon(Bool_t open=kFALSE);
   virtual const TGPicture* GetListTreeCheckBoxIcon();

   ClassDef(TEveElement, 0); // Base class for TEveUtil visualization elements, providing hierarchy management, rendering control and list-tree item management.
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

   virtual TObject* GetObject(const TEveException& eh="TEveElementObjectPtr::GetObject ") const;
   virtual void     ExportToCINT(Text_t* var_name);

   Bool_t GetOwnObject() const   { return fOwnObject; }
   void   SetOwnObject(Bool_t o) { fOwnObject = o; }

   ClassDef(TEveElementObjectPtr, 0); // TEveElement with external TObject as a holder of visualization data.
};


/******************************************************************************/
// TEveElementList
/******************************************************************************/

class TEveElementList : public TEveElement,
                        public TNamed
{
private:
   TEveElementList(const TEveElementList&);            // Not implemented
   TEveElementList& operator=(const TEveElementList&); // Not implemented

protected:
   Color_t   fColor;       // Color of the object.
   Bool_t    fDoColor;     // Should serve fColor as the main color of the object.
   TClass   *fChildClass;  // Class of acceptable children, others are rejected.

public:
   TEveElementList(const Text_t* n="TEveElementList", const Text_t* t="",
                   Bool_t doColor=kFALSE);
   virtual ~TEveElementList() {}

   virtual const Text_t* GetElementName()  const { return TNamed::GetName(); }
   virtual const Text_t* GetElementTitle() const { return TNamed::GetTitle(); }
   virtual void SetElementName (const Text_t* name)  { TNamed::SetName(name); }
   virtual void SetElementTitle(const Text_t* title) { TNamed::SetTitle(title); }
   virtual void SetElementNameTitle(const Text_t* name, const Text_t* title)
   { TNamed::SetNameTitle(name, title); }

   virtual Bool_t CanEditMainColor() const { return fDoColor; }

   TClass* GetChildClass() const { return fChildClass; }
   void SetChildClass(TClass* c) { fChildClass = c; }

   virtual Bool_t AcceptElement(TEveElement* el);

   ClassDef(TEveElementList, 0); // List of TEveElement objects with a possibility to limit the class of accepted elements.
};

#endif
