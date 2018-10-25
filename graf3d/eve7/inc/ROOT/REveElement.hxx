// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveElement_hxx
#define ROOT7_REveElement_hxx

#include <ROOT/REveUtil.hxx>
#include <ROOT/REveVector.hxx>
#include <ROOT/REveProjectionBases.hxx>

#include "TNamed.h"
#include "TRef.h"
#include <memory>

class TGeoMatrix;

/// use temporary solution for forwarding of nlohmann::json
/// after version of 3.1.0 it is included in official releases
/// see https://github.com/nlohmann/json/issues/314

#include <cstdint> // int64_t, uint64_t
#include <map> // map
#include <memory> // allocator
#include <string> // string
#include <vector> // vector

namespace nlohmann {

  template<typename T, typename SFINAE>
    struct adl_serializer;

  template<template<typename U, typename V, typename... Args> class ObjectType,
           template<typename U, typename... Args> class ArrayType,
           class StringType,
           class BooleanType,
           class NumberIntegerType,
           class NumberUnsignedType,
           class NumberFloatType,
           template<typename U> class AllocatorType,
           template<typename T, typename SFINAE> class JSONSerializer>
     class basic_json;

   using json = basic_json<std::map, std::vector, std::string, bool, std::int64_t, std::uint64_t, double, std::allocator, adl_serializer>;
}

namespace ROOT {
namespace Experimental {

typedef unsigned int ElementId_t;

class REveScene;
class REveCompound;
class REveTrans;
class REveRenderData;

/******************************************************************************/
// REveElement
/******************************************************************************/

class REveElement
{
   friend class REveManager;
   friend class REveScene;

   REveElement& operator=(const REveElement&); // Not implemented

public:
   typedef std::list<REveElement*>              List_t;
   typedef List_t::iterator                     List_i;
   typedef List_t::const_iterator               List_ci;

   typedef std::set<REveElement*>               Set_t;
   typedef Set_t::iterator                      Set_i;
   typedef Set_t::const_iterator                Set_ci;

private:
   ElementId_t      fElementId{0};        // Unique ID of an element.

protected:
   REveElement     *fMother{nullptr};
   REveScene       *fScene{nullptr};

   ElementId_t get_mother_id() const;
   ElementId_t get_scene_id()  const;

   void assign_element_id_recurisvely();
   void assign_scene_recursively(REveScene* s);

protected:
   List_t           fParents;              //  List of parents.
   List_t           fChildren;             //  List of children.
   REveCompound    *fCompound{nullptr};    //  Compound this object belongs to.
   REveElement     *fVizModel{nullptr};    //! Element used as model from VizDB.
   TString          fVizTag;               //  Tag used to query VizDB for model element.

   Int_t            fNumChildren;          //!
   Int_t            fParentIgnoreCnt;      //! Counter for parents that are ignored in ref-counting.
   Int_t            fDenyDestroy;          //! Deny-destroy count.
   Bool_t           fDestroyOnZeroRefCnt;  //  Auto-destruct when ref-count reaches zero.

   Bool_t           fRnrSelf;                 //  Render this element.
   Bool_t           fRnrChildren;             //  Render children of this element.
   Bool_t           fCanEditMainColor;        //  Allow editing of main color.
   Bool_t           fCanEditMainTransparency; //  Allow editing of main transparency.
   Bool_t           fCanEditMainTrans;        //  Allow editing of main transformation.

   Char_t           fMainTransparency;      //  Main-transparency variable.
   Color_t         *fMainColorPtr{nullptr}; //  Pointer to main-color variable.
   std::unique_ptr<REveTrans> fMainTrans;   //  Pointer to main transformation matrix.

   TRef             fSource;               //  External object that is represented by this element.
   void            *fUserData{nullptr};    //! Externally assigned and controlled user data.
   std::unique_ptr<REveRenderData> fRenderData;//! Vertex / normal / triangle index information for rendering.

   virtual void PreDeleteElement();
   virtual void RemoveElementsInternal();
   virtual void AnnihilateRecursively();

   static const char* ToString(Bool_t b);

public:
   REveElement();
   REveElement(Color_t& main_color);
   REveElement(const REveElement& e);
   virtual ~REveElement();

   ElementId_t GetElementId() const { return fElementId; }

   virtual REveElement* CloneElement() const;
   virtual REveElement* CloneElementRecurse(Int_t level=0) const;
   virtual void         CloneChildrenRecurse(REveElement* dest, Int_t level=0) const;

   virtual const char* GetElementName()  const;
   virtual const char* GetElementTitle() const;
   virtual TString     GetHighlightTooltip() { return TString(GetElementTitle()); }
   virtual void SetElementName (const char* name);
   virtual void SetElementTitle(const char* title);
   virtual void SetElementNameTitle(const char* name, const char* title);
   virtual void NameTitleChanged();

   const TString& GetVizTag() const             { return fVizTag; }
   void           SetVizTag(const TString& tag) { fVizTag = tag;  }

   REveElement*   GetVizModel() const           { return fVizModel; }
   void           SetVizModel(REveElement* model);
   Bool_t         FindVizModel();

   Bool_t         ApplyVizTag(const TString& tag, const TString& fallback_tag="");

   virtual void PropagateVizParamsToProjecteds();
   virtual void PropagateVizParamsToElements(REveElement* el=0);
   virtual void CopyVizParams(const REveElement* el);
   virtual void CopyVizParamsFromDB();
   void         SaveVizParams (std::ostream& out, const TString& tag, const TString& var);
   virtual void WriteVizParams(std::ostream& out, const TString& var);

   REveElement*  GetMaster();
   REveCompound* GetCompound()                { return fCompound; }
   void          SetCompound(REveCompound* c) { fCompound = c;    }

   REveScene*   GetScene()  { return fScene;  }
   REveElement* GetMother() { return fMother; }

   virtual void AddParent(REveElement* el);
   virtual void RemoveParent(REveElement* el);
   virtual void CheckReferenceCount(const REveException& eh="REveElement::CheckReferenceCount ");
   virtual void CollectSceneParents(List_t& scenes);
   virtual void CollectSceneParentsFromChildren(List_t& scenes,
                                                REveElement* parent);

   List_i  BeginParents()        { return  fParents.begin();  }
   List_i  EndParents()          { return  fParents.end();    }
   List_ci BeginParents()  const { return  fParents.begin();  }
   List_ci EndParents()    const { return  fParents.end();    }
   Int_t   NumParents()    const { return  fParents.size();   }
   Bool_t  HasParents()    const { return !fParents.empty();  }

   const List_t& RefChildren() const { return  fChildren;     }
   List_i  BeginChildren()       { return  fChildren.begin(); }
   List_i  EndChildren()         { return  fChildren.end();   }
   List_ci BeginChildren() const { return  fChildren.begin(); }
   List_ci EndChildren()   const { return  fChildren.end();   }
   Int_t   NumChildren()   const { return  fNumChildren;      }
   Bool_t  HasChildren()   const { return  fNumChildren != 0; }

   Bool_t       HasChild(REveElement *el);
   REveElement *FindChild(const TString &name, const TClass *cls = nullptr);
   REveElement *FindChild(TPRegexp &regexp, const TClass *cls = nullptr);
   Int_t        FindChildren(List_t &matches, const TString&  name, const TClass *cls = nullptr);
   Int_t        FindChildren(List_t &matches, TPRegexp& regexp, const TClass* cls = nullptr);
   REveElement *FirstChild() const;
   REveElement *LastChild () const;

   void EnableListElements(Bool_t rnr_self = kTRUE, Bool_t rnr_children = kTRUE);    // *MENU*
   void DisableListElements(Bool_t rnr_self = kFALSE, Bool_t rnr_children = kFALSE); // *MENU*

   Bool_t GetDestroyOnZeroRefCnt() const;
   void   SetDestroyOnZeroRefCnt(Bool_t d);

   // TODO: this is first candidate to be replaced by shared_ptr
   Int_t  GetDenyDestroy() const;
   void   IncDenyDestroy();
   void   DecDenyDestroy();

   Int_t  GetParentIgnoreCnt() const;
   void   IncParentIgnoreCnt();
   void   DecParentIgnoreCnt();

   virtual TObject* GetObject      (const REveException& eh) const;
   virtual TObject* GetEditorObject(const REveException& eh) const { return GetObject(eh); }
   virtual TObject* GetRenderObject(const REveException& eh) const { return GetObject(eh); }

   // --------------------------------

   virtual void ExportToCINT(char *var_name); // *MENU*

   void    DumpSourceObject() const;                       // *MENU*
   void    PrintSourceObject() const;                      // *MENU*
   void    ExportSourceObjectToCINT(char* var_name) const; // *MENU*

   virtual Bool_t AcceptElement(REveElement* el);

   virtual void AddElement(REveElement* el);
   virtual void RemoveElement(REveElement* el);
   virtual void RemoveElementLocal(REveElement* el);
   virtual void RemoveElements();
   virtual void RemoveElementsLocal();

   virtual void AnnihilateElements();
   virtual void Annihilate();

   virtual void ProjectChild(REveElement* el, Bool_t same_depth=kTRUE);
   virtual void ProjectAllChildren(Bool_t same_depth=kTRUE);

   virtual void Destroy();                      // *MENU*
   virtual void DestroyOrWarn();
   virtual void DestroyElements();              // *MENU*

   virtual Bool_t HandleElementPaste(REveElement* el);
   virtual void   ElementChanged(Bool_t update_scenes=kTRUE, Bool_t redraw=kFALSE);

   virtual Bool_t CanEditElement() const { return kTRUE; }
   virtual Bool_t SingleRnrState() const { return kFALSE; }
   virtual Bool_t GetRnrSelf()     const { return fRnrSelf; }
   virtual Bool_t GetRnrChildren() const { return fRnrChildren; }
   virtual Bool_t GetRnrState()    const { return fRnrSelf && fRnrChildren; }
   virtual Bool_t GetRnrAnything() const { return fRnrSelf || (fRnrChildren && HasChildren()); }
   virtual Bool_t SetRnrSelf(Bool_t rnr);
   virtual Bool_t SetRnrChildren(Bool_t rnr);
   virtual Bool_t SetRnrSelfChildren(Bool_t rnr_self, Bool_t rnr_children);
   virtual Bool_t SetRnrState(Bool_t rnr);
   virtual void   PropagateRnrStateToProjecteds();

   virtual Bool_t CanEditMainColor() const   { return fCanEditMainColor; }
   void           SetEditMainColor(Bool_t x) { fCanEditMainColor = x; }
   Color_t* GetMainColorPtr()        const   { return fMainColorPtr; }
   void     SetMainColorPtr(Color_t* color)  { fMainColorPtr = color; }

   virtual Bool_t  HasMainColor() const { return fMainColorPtr != nullptr; }
   virtual Color_t GetMainColor() const { return fMainColorPtr ? *fMainColorPtr : 0; }
   virtual void    SetMainColor(Color_t color);
   void            SetMainColorPixel(Pixel_t pixel);
   void            SetMainColorRGB(UChar_t r, UChar_t g, UChar_t b);
   void            SetMainColorRGB(Float_t r, Float_t g, Float_t b);
   virtual void    PropagateMainColorToProjecteds(Color_t color, Color_t old_color);

   virtual Bool_t  CanEditMainTransparency() const   { return fCanEditMainTransparency; }
   void            SetEditMainTransparency(Bool_t x) { fCanEditMainTransparency = x; }
   virtual Char_t  GetMainTransparency()     const { return fMainTransparency; }
   virtual void    SetMainTransparency(Char_t t);
   void            SetMainAlpha(Float_t alpha);
   virtual void    PropagateMainTransparencyToProjecteds(Char_t t, Char_t old_t);

   virtual Bool_t     CanEditMainTrans() const { return fCanEditMainTrans; }
   virtual Bool_t     HasMainTrans()     const { return fMainTrans.get() != nullptr;   }
   virtual REveTrans* PtrMainTrans(Bool_t create=kTRUE);
   virtual REveTrans& RefMainTrans();
   virtual void       InitMainTrans(Bool_t can_edit=kTRUE);
   virtual void       DestroyMainTrans();

   virtual void SetTransMatrix(Double_t *carr);
   virtual void SetTransMatrix(const TGeoMatrix &mat);

   virtual Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset);
   virtual void  BuildRenderData();

   TRef&    GetSource()                 { return fSource; }
   TObject* GetSourceObject()     const { return fSource.GetObject(); }
   void     SetSourceObject(TObject *o) { fSource = o; }

   void* GetUserData() const { return fUserData; }
   void  SetUserData(void* ud) { fUserData = ud; }

   REveRenderData *GetRenderData() const { return fRenderData.get(); }


   // Selection state and management
   //--------------------------------

protected:
   Bool_t  fPickable;
   Bool_t  fSelected;             //!
   Bool_t  fHighlighted;          //!
   Short_t fImpliedSelected;      //!
   Short_t fImpliedHighlighted;   //!

   enum ECompoundSelectionColorBits
   {
      kCSCBImplySelectAllChildren           = BIT(0), // compound will select all children
      kCSCBTakeAnyParentAsMaster            = BIT(1), // element will take any compound parent as master
      kCSCBApplyMainColorToAllChildren      = BIT(2), // compound will apply color change to all children
      kCSCBApplyMainColorToMatchingChildren = BIT(3), // compound will apply color change to all children with matching color
      kCSCBApplyMainTransparencyToAllChildren      = BIT(4), // compound will apply transparency change to all children
      kCSCBApplyMainTransparencyToMatchingChildren = BIT(5)  // compound will apply transparency change to all children with matching color
   };

   enum EDestruct
   {
      kNone,
      kStandard,
      kAnnihilate
   };

   UChar_t fCSCBits;

public:
   typedef void (REveElement::* Select_foo)      (Bool_t);
   typedef void (REveElement::* ImplySelect_foo) ();

   Bool_t IsPickable()    const { return fPickable; }
   void   SetPickable(Bool_t p) { fPickable = p; }
   void   SetPickableRecursively(Bool_t p);

   virtual REveElement* ForwardSelection();
   virtual REveElement* ForwardEdit();

   virtual void SelectElement(Bool_t state);
   virtual void IncImpliedSelected();
   virtual void DecImpliedSelected();
   virtual void UnSelected();

   virtual void HighlightElement(Bool_t state);
   virtual void IncImpliedHighlighted();
   virtual void DecImpliedHighlighted();
   virtual void UnHighlighted();

   virtual void FillImpliedSelectedSet(Set_t& impSelSet);

   virtual UChar_t GetSelectedLevel() const;

   void   RecheckImpliedSelections();

   void   SetCSCBits(UChar_t f)   { fCSCBits |=  f; }
   void   ResetCSCBits(UChar_t f) { fCSCBits &= ~f; }
   Bool_t TestCSCBits(UChar_t f) const { return (fCSCBits & f) != 0; }

   void   ResetAllCSCBits()                     { fCSCBits  =  0; }
   void   CSCImplySelectAllChildren()           { fCSCBits |= kCSCBImplySelectAllChildren; }
   void   CSCTakeAnyParentAsMaster()            { fCSCBits |= kCSCBTakeAnyParentAsMaster;  }
   void   CSCApplyMainColorToAllChildren()      { fCSCBits |= kCSCBApplyMainColorToAllChildren; }
   void   CSCApplyMainColorToMatchingChildren() { fCSCBits |= kCSCBApplyMainColorToMatchingChildren; }
   void   CSCApplyMainTransparencyToAllChildren()      { fCSCBits |= kCSCBApplyMainTransparencyToAllChildren; }
   void   CSCApplyMainTransparencyToMatchingChildren() { fCSCBits |= kCSCBApplyMainTransparencyToMatchingChildren; }


   // Change-stamping and change bits
   //---------------------------------

   enum EChangeBits
   {
      kCBColorSelection =  BIT(0), // Main color or select/hilite state changed.
      kCBTransBBox      =  BIT(1), // Transformation matrix or bounding-box changed.
      kCBObjProps       =  BIT(2), // Object changed, requires dropping its display-lists.
      kCBVisibility     =  BIT(3)  // Rendering of self/children changed.
      // kCBElementAdded   = BIT(), // Element was added to a new parent.
      // kCBElementRemoved = BIT()  // Element was removed from a parent.

      // Deletions are handled in a special way in REveManager::PreDeleteElement().
   };

protected:
   UChar_t      fChangeBits;  //!
   Char_t       fDestructing; //!

public:
   void StampColorSelection() { AddStamp(kCBColorSelection); }
   void StampTransBBox()      { AddStamp(kCBTransBBox); }
   void StampObjProps()       { AddStamp(kCBObjProps); }
   void StampVisibility()     { AddStamp(kCBVisibility); }
   // void StampElementAdded()   { AddStamp(kCBElementAdded); }
   // void StampElementRemoved() { AddStamp(kCBElementRemoved); }
   virtual void AddStamp(UChar_t bits);
   virtual void ClearStamps() { fChangeBits = 0; }

   UChar_t GetChangeBits() const { return fChangeBits; }


   // Menu entries for VizDB communication (here so they are last in the menu).

   void VizDB_Apply(const char* tag);           // *MENU*
   void VizDB_Reapply();                        // *MENU*
   void VizDB_UpdateModel(Bool_t update=kTRUE); // *MENU*
   void VizDB_Insert(const char* tag, Bool_t replace=kTRUE, Bool_t update=kTRUE); // *MENU*

   ClassDef(REveElement, 0); // Base class for REveUtil visualization elements, providing hierarchy management, rendering control and list-tree item management.
};


/******************************************************************************/
// REveElementObjectPtr
// FIXME: Not used, can be removed?
/******************************************************************************/

class REveElementObjectPtr : public REveElement,
                             public TObject
{
   REveElementObjectPtr& operator=(const REveElementObjectPtr&); // Not implemented

protected:
   TObject* fObject{nullptr};     // External object holding the visual data.
   Bool_t   fOwnObject{kFALSE};  // Is object owned / should be deleted on destruction.

public:
   REveElementObjectPtr(TObject* obj, Bool_t own=kTRUE);
   REveElementObjectPtr(TObject* obj, Color_t& mainColor, Bool_t own=kTRUE);
   REveElementObjectPtr(const REveElementObjectPtr& e);
   virtual ~REveElementObjectPtr();

   virtual REveElementObjectPtr* CloneElement() const;

   virtual TObject* GetObject(const REveException& eh="REveElementObjectPtr::GetObject ") const;
   virtual void     ExportToCINT(char* var_name);

   Bool_t GetOwnObject() const   { return fOwnObject; }
   void   SetOwnObject(Bool_t o) { fOwnObject = o; }

   ClassDef(REveElementObjectPtr, 0); // REveElement with external TObject as a holder of visualization data.
};


/******************************************************************************/
// REveElementList
/******************************************************************************/

class REveElementList : public REveElement,
                        public TNamed,
                        public REveProjectable
{
private:
   REveElementList& operator=(const REveElementList&); // Not implemented

protected:
   Color_t   fColor{0};              // Color of the object.
   TClass   *fChildClass{nullptr};   // Class of acceptable children, others are rejected.

public:
   REveElementList(const char *n = "REveElementList", const char *t = "", Bool_t doColor = kFALSE,
                   Bool_t doTransparency = kFALSE);
   REveElementList(const REveElementList &e);
   virtual ~REveElementList() {}

   virtual TObject* GetObject(const REveException& /*eh*/="REveElementList::GetObject ") const
   { const TObject* obj = this; return const_cast<TObject*>(obj); }

   virtual REveElementList* CloneElement() const;

   virtual const char* GetElementName()  const { return GetName();  }
   virtual const char* GetElementTitle() const { return GetTitle(); }

   virtual void SetElementName (const char* name)
   { TNamed::SetName(name); NameTitleChanged(); }

   virtual void SetElementTitle(const char* title)
   { TNamed::SetTitle(title); NameTitleChanged(); }

   virtual void SetElementNameTitle(const char* name, const char* title)
   { TNamed::SetNameTitle(name, title); NameTitleChanged(); }

   TClass* GetChildClass() const { return fChildClass; }
   void    SetChildClass(TClass* c) { fChildClass = c; }

   // Element
   Bool_t  AcceptElement(REveElement* el); // override;

   // Projectable
   TClass* ProjectedClass(const REveProjection* p) const; // override;

   ClassDef(REveElementList, 0); // List of REveElement objects with a possibility to limit the class of accepted elements.
};


/******************************************************************************/
// REveElementListProjected
/******************************************************************************/

class REveElementListProjected : public REveElementList,
                                 public REveProjected
{
private:
   REveElementListProjected(const REveElementListProjected&);            // Not implemented
   REveElementListProjected& operator=(const REveElementListProjected&); // Not implemented

public:
   REveElementListProjected();
   virtual ~REveElementListProjected() {}

   virtual void UpdateProjection();
   virtual REveElement* GetProjectedAsElement() { return this; }

   ClassDef(REveElementListProjected, 0); // Projected REveElementList.
};

} // namespace Experimental
} // namespace Experimental

#endif
